"""
Live MLEvolve experiment dashboard.
Usage: .venv/bin/python dashboard.py [--port 8050]
"""
import gzip
import json
import argparse
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.after_request
def compress(response):
    if 'gzip' not in request.headers.get('Accept-Encoding', ''):
        return response
    if response.status_code < 200 or response.status_code >= 300:
        return response
    if response.content_type and 'application/json' in response.content_type:
        response.data = gzip.compress(response.data, compresslevel=1)
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = len(response.data)
    return response
RUNS_DIR = Path("runs")

# Simple file cache: path -> (mtime, data)
_file_cache = {}


def _cached_json(path):
    """Read and cache a JSON file, re-reading only when mtime changes."""
    path = Path(path)
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    cached = _file_cache.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    with open(path) as f:
        data = json.load(f)
    _file_cache[str(path)] = (mtime, data)
    return data


def find_runs():
    """Find all run directories, newest first."""
    runs = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        journal = d / "logs" / "journal.json"
        if journal.exists():
            runs.append({"name": d.name, "path": str(d)})
    return runs


def load_journal(run_name):
    journal_path = RUNS_DIR / run_name / "logs" / "journal.json"
    return _cached_json(journal_path)


def load_best_code(run_name):
    code_path = RUNS_DIR / run_name / "logs" / "best_solution.py"
    if not code_path.exists():
        return None
    return code_path.read_text()


def load_config(run_name):
    cfg_path = RUNS_DIR / run_name / "logs" / "config.yaml"
    if not cfg_path.exists():
        return None
    return cfg_path.read_text()


def load_log_tail(run_name, lines=80):
    log_path = RUNS_DIR / run_name / "logs" / "MLEvolve.log"
    if not log_path.exists():
        return None
    all_lines = log_path.read_text().splitlines()
    return "\n".join(all_lines[-lines:])


def load_socrates_transcripts(run_name):
    path = RUNS_DIR / run_name / "logs" / "socrates_transcripts.jsonl"
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def get_exp_id(run_name):
    cfg_text = load_config(run_name)
    if not cfg_text:
        return None
    for line in cfg_text.splitlines():
        if line.startswith("exp_id:"):
            return line.split(":", 1)[1].strip()
    return None


def get_dataset_dir(run_name):
    cfg_text = load_config(run_name)
    if not cfg_text:
        return None
    lines = cfg_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("dataset_dir:"):
            value_part = line.split(":", 1)[1].strip()
            if value_part and not value_part.startswith("!!"):
                return value_part
            parts = []
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("- "):
                    parts.append(lines[j][2:].strip())
                else:
                    break
            parts = [p for p in parts if p]
            if parts:
                return str(Path(*parts))
    return None


def load_test_scores(run_name):
    path = RUNS_DIR / run_name / "logs" / "test_scores.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def grade_run(run_name):
    try:
        from mlebench.grade import grade_csv
        from mlebench.registry import registry
    except ImportError:
        return {"error": "mlebench is not installed"}

    exp_id = get_exp_id(run_name)
    dataset_dir = get_dataset_dir(run_name)
    if not exp_id or not dataset_dir:
        return {"error": "Missing exp_id or dataset_dir in config"}

    comp = registry.set_data_dir(Path(dataset_dir)).get_competition(exp_id)

    sub_dir = RUNS_DIR / run_name / "workspace" / "submission"
    if not sub_dir.exists():
        return {"error": "No submission directory found"}

    scores = {}
    thresholds = None
    is_lower_better = None

    for csv_path in sorted(sub_dir.glob("submission_*.csv")):
        node_id = csv_path.stem.replace("submission_", "")
        short_id = node_id[:8]
        try:
            report = grade_csv(csv_path, comp)
        except Exception:
            scores[short_id] = {"score": None, "medal": None}
            continue

        if thresholds is None:
            thresholds = {
                "gold": report.gold_threshold,
                "silver": report.silver_threshold,
                "bronze": report.bronze_threshold,
                "median": report.median_threshold,
            }
            is_lower_better = report.is_lower_better

        if report.valid_submission and report.score is not None:
            medal = ("GOLD" if report.gold_medal else
                     "SILVER" if report.silver_medal else
                     "BRONZE" if report.bronze_medal else
                     "ABOVE MEDIAN" if report.above_median else "BELOW")
            scores[short_id] = {"score": report.score, "medal": medal}
        else:
            scores[short_id] = {"score": None, "medal": None}

    result = {
        "exp_id": exp_id,
        "is_lower_better": is_lower_better,
        "thresholds": thresholds,
        "scores": scores,
        "graded_at": datetime.now().isoformat(),
    }

    cache_path = RUNS_DIR / run_name / "logs" / "test_scores.json"
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


@app.route("/")
def index():
    return DASHBOARD_HTML


@app.route("/api/runs")
def api_runs():
    return jsonify(find_runs())


@app.route("/api/run/<run_name>")
def api_run(run_name):
    journal = load_journal(run_name)
    if journal is None:
        return jsonify({"error": "not found"}), 404

    nodes = journal["nodes"]
    node2parent = journal.get("node2parent", {})

    summary_nodes = []
    for n in nodes:
        summary_nodes.append({
            "step": n["step"],
            "id": n["id"][:8],
            "stage": n["stage"],
            "metric": n["metric"]["value"],
            "maximize": n["metric"]["maximize"],
            "is_buggy": n["is_buggy"],
            "exec_time": n.get("exec_time"),
            "created_time": n.get("created_time"),
            "finish_time": n.get("finish_time"),
            "plan": (n.get("plan") or "")[:300],
            "parent": node2parent.get(n["id"], None),
        })

    maximize = None
    for n in nodes:
        if n["metric"]["maximize"] is not None:
            maximize = n["metric"]["maximize"]
            break

    best_progression = []
    current_best = None
    for n in sorted(nodes, key=lambda x: x["step"]):
        v = n["metric"]["value"]
        if v is None:
            continue
        if current_best is None:
            current_best = v
        elif maximize and v > current_best:
            current_best = v
        elif not maximize and v < current_best:
            current_best = v
        best_progression.append({"step": n["step"], "best": current_best, "value": v})

    stages = {}
    for n in nodes:
        s = n["stage"]
        if s == "root":
            continue
        stages[s] = stages.get(s, 0) + 1

    buggy = sum(1 for n in nodes if n["is_buggy"] is True)
    good = sum(1 for n in nodes if n["is_buggy"] is False)
    pending = sum(1 for n in nodes if n["is_buggy"] is None and n["stage"] != "root")

    total_steps_cfg = None
    cfg_text = load_config(run_name)
    if cfg_text:
        for line in cfg_text.splitlines():
            if "steps:" in line and "initial" not in line:
                parts = line.split(":")
                if len(parts) == 2:
                    val = parts[1].strip()
                    if val.isdigit():
                        total_steps_cfg = int(val)
                        break

    test_scores = load_test_scores(run_name)

    return jsonify({
        "run_name": run_name,
        "total_nodes": len(nodes) - 1,
        "total_steps_cfg": total_steps_cfg,
        "maximize": maximize,
        "nodes": summary_nodes,
        "best_progression": best_progression,
        "stages": stages,
        "buggy": buggy,
        "good": good,
        "pending": pending,
        "test_scores": test_scores,
    })


@app.route("/api/run/<run_name>/node/<node_id>")
def api_node_detail(run_name, node_id):
    """Return full detail for a single node (lazy-loaded on double-click)."""
    journal = load_journal(run_name)
    if journal is None:
        return jsonify({"error": "not found"}), 404
    for n in journal["nodes"]:
        if n["id"][:8] == node_id or n["id"] == node_id:
            return jsonify({
                "step": n["step"],
                "id": n["id"][:8],
                "stage": n["stage"],
                "metric": n["metric"]["value"],
                "is_buggy": n["is_buggy"],
                "exec_time": n.get("exec_time"),
                "plan": n.get("plan") or "",
                "analysis": n.get("analysis") or "",
                "code_summary": n.get("code_summary") or "",
                "term_out": n.get("_term_out") or [],
                "exc_type": n.get("exc_type"),
                "exc_info": n.get("exc_info"),
                "code": n.get("code") or "",
            })
    return jsonify({"error": "node not found"}), 404


@app.route("/api/run/<run_name>/code-logs")
def api_code_logs(run_name):
    """Return best code and log tail (lazy-loaded for Code & Logs tab)."""
    return jsonify({
        "best_code": load_best_code(run_name),
        "log_tail": load_log_tail(run_name),
    })


@app.route("/api/run/<run_name>/socrates")
def api_socrates(run_name):
    """Return Socrates transcripts (lazy-loaded for Socrates tab)."""
    return jsonify(load_socrates_transcripts(run_name))


@app.route("/api/run/<run_name>/grade", methods=["POST"])
def api_grade(run_name):
    result = grade_run(run_name)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLEvolve Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/styles/github-dark.min.css">
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/languages/python.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script>marked.use({async: false});</script>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text2: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --orange: #d29922; --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, 'SF Mono', 'Menlo', monospace; background: var(--bg); color: var(--text); font-size: 13px; }
  .header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 10; }
  .header h1 { font-size: 16px; font-weight: 600; }
  .header select { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-size: 13px; }
  .header .status { margin-left: auto; display: flex; align-items: center; gap: 8px; }
  .header .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .header .refresh-info { color: var(--text2); font-size: 11px; }

  .grade-btn { background: transparent; color: var(--orange); border: 1px solid var(--orange); border-radius: 6px; padding: 4px 12px; font-size: 12px; cursor: pointer; font-weight: 600; font-family: inherit; }
  .grade-btn:hover { background: rgba(210,153,34,0.1); }
  .grade-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* Nav tabs */
  .nav-tabs { display: flex; gap: 0; background: var(--surface); border-bottom: 1px solid var(--border); padding: 0 20px; }
  .nav-tab { padding: 10px 20px; cursor: pointer; color: var(--text2); font-size: 13px; font-weight: 500; border-bottom: 2px solid transparent; transition: all 0.2s; }
  .nav-tab:hover { color: var(--text); }
  .nav-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .page { display: none; }
  .page.active { display: block; }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 16px; max-width: 1400px; margin: 0 auto; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .card h2 { font-size: 12px; text-transform: uppercase; color: var(--text2); letter-spacing: 0.5px; margin-bottom: 10px; }
  .card.full { grid-column: 1 / -1; }

  .kpi-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .kpi { background: var(--bg); border-radius: 6px; padding: 10px 14px; flex: 1; min-width: 120px; }
  .kpi .label { font-size: 10px; text-transform: uppercase; color: var(--text2); letter-spacing: 0.5px; }
  .kpi .value { font-size: 22px; font-weight: 700; margin-top: 2px; }
  .kpi .value.green { color: var(--green); }
  .kpi .value.red { color: var(--red); }
  .kpi .value.accent { color: var(--accent); }
  .kpi .value.orange { color: var(--orange); }

  .chart-box { position: relative; height: 220px; }

  .node-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .node-table th { text-align: left; color: var(--text2); font-weight: 500; padding: 6px 8px; border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--surface); }
  .node-table td { padding: 5px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
  .node-table tr:hover { background: rgba(88,166,255,0.05); }
  .table-scroll { max-height: 350px; overflow-y: auto; }

  .badge { display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 10px; font-weight: 600; }
  .badge.draft { background: rgba(88,166,255,0.15); color: var(--accent); }
  .badge.improve { background: rgba(63,185,80,0.15); color: var(--green); }
  .badge.debug { background: rgba(248,81,73,0.15); color: var(--red); }
  .badge.evolution { background: rgba(188,140,255,0.15); color: var(--purple); }
  .badge.fusion, .badge.fusion_draft { background: rgba(210,153,34,0.15); color: var(--orange); }
  .badge.root { background: rgba(139,148,158,0.15); color: var(--text2); }

  .medal-badge { display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 10px; font-weight: 700; vertical-align: middle; }
  .medal-badge.gold { background: rgba(255,215,0,0.2); color: #FFD700; }
  .medal-badge.silver { background: rgba(192,192,192,0.2); color: #C0C0C0; }
  .medal-badge.bronze { background: rgba(205,127,50,0.2); color: #CD7F32; }
  .medal-badge.above-median { background: rgba(63,185,80,0.2); color: var(--green); }
  .medal-badge.below { background: rgba(248,81,73,0.2); color: var(--red); }

  .buggy-true { color: var(--red); }
  .buggy-false { color: var(--green); }

  .code-wrapper { position: relative; }
  .copy-btn { position: absolute; top: 8px; right: 8px; background: var(--border); color: var(--text2); border: none; border-radius: 4px; padding: 4px 10px; font-size: 11px; cursor: pointer; font-family: inherit; z-index: 1; }
  .copy-btn:hover { background: var(--accent); color: var(--bg); }
  pre.code-block { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-size: 11px; max-height: 400px; overflow: auto; white-space: pre-wrap; word-break: break-all; line-height: 1.5; }
  pre.code-block code.hljs { background: transparent; padding: 0; }
  pre.log-block { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-size: 11px; max-height: 300px; overflow: auto; white-space: pre-wrap; word-break: break-all; color: var(--text2); line-height: 1.4; }

  .plan-text { max-width: 400px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: pointer; }
  .plan-text:hover { white-space: normal; overflow: visible; }
  .node-table tr.selected { background: rgba(88,166,255,0.12); }

  /* Step detail page */
  .step-detail { max-width: 1000px; margin: 0 auto; padding: 16px; }
  .step-detail .detail-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
  .step-detail .detail-header h2 { font-size: 14px; color: var(--accent); }
  .step-detail .detail-meta { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; font-size: 12px; color: var(--text2); }
  .step-detail .detail-meta span { background: var(--bg); padding: 4px 10px; border-radius: 6px; }
  .step-detail .plan-content { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; line-height: 1.7; font-size: 13px; }
  .step-detail .plan-content h1, .step-detail .plan-content h2, .step-detail .plan-content h3 { color: var(--accent); margin: 16px 0 8px; }
  .step-detail .plan-content h1 { font-size: 18px; }
  .step-detail .plan-content h2 { font-size: 15px; }
  .step-detail .plan-content h3 { font-size: 13px; }
  .step-detail .plan-content p { margin: 8px 0; }
  .step-detail .plan-content code { background: var(--bg); padding: 1px 5px; border-radius: 3px; font-size: 12px; }
  .step-detail .plan-content pre { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 12px; overflow-x: auto; margin: 8px 0; }
  .step-detail .plan-content pre code { background: none; padding: 0; }
  .step-detail .plan-content ul, .step-detail .plan-content ol { margin: 8px 0; padding-left: 24px; }
  .step-detail .plan-content li { margin: 4px 0; }
  .step-detail .back-btn { background: transparent; color: var(--accent); border: 1px solid var(--accent); border-radius: 6px; padding: 4px 12px; font-size: 12px; cursor: pointer; font-family: inherit; }
  .step-detail .back-btn:hover { background: rgba(88,166,255,0.1); }

  /* Socrates styles */
  .socrates-list { max-width: 1400px; margin: 0 auto; padding: 16px; display: flex; flex-direction: column; gap: 16px; }
  .socrates-session { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .socrates-session-header { padding: 12px 16px; display: flex; align-items: center; gap: 12px; cursor: pointer; border-bottom: 1px solid var(--border); }
  .socrates-session-header:hover { background: rgba(88,166,255,0.03); }
  .socrates-session-header .session-num { font-weight: 700; color: var(--accent); min-width: 30px; }
  .socrates-session-header .session-time { color: var(--text2); font-size: 11px; }
  .socrates-session-header .session-plan { color: var(--text); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 12px; }
  .socrates-session-header .approved-badge { padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 600; }
  .socrates-session-header .approved-badge.yes { background: rgba(63,185,80,0.15); color: var(--green); }
  .socrates-session-header .approved-badge.no { background: rgba(248,81,73,0.15); color: var(--red); }
  .socrates-session-header .rounds-info { color: var(--text2); font-size: 11px; }
  .socrates-session-header .chevron { color: var(--text2); transition: transform 0.2s; }
  .socrates-session.open .chevron { transform: rotate(90deg); }
  .socrates-session-body { display: none; padding: 0; }
  .socrates-session.open .socrates-session-body { display: block; }
  .socrates-round { border-bottom: 1px solid var(--border); }
  .socrates-round:last-child { border-bottom: none; }
  .round-label { padding: 8px 16px; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text2); background: var(--bg); }
  .socrates-msg { padding: 12px 16px; line-height: 1.6; font-size: 12px; white-space: pre-wrap; word-break: break-word; }
  .socrates-msg.socrates { border-left: 3px solid var(--purple); background: rgba(188,140,255,0.03); }
  .socrates-msg.planner { border-left: 3px solid var(--accent); background: rgba(88,166,255,0.03); }
  .msg-role { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 6px; }
  .msg-role.socrates { color: var(--purple); }
  .msg-role.planner { color: var(--accent); }
  .no-socrates { color: var(--text2); text-align: center; padding: 60px 20px; font-size: 14px; }
</style>
</head>
<body>

<div class="header">
  <h1>MLEvolve Dashboard</h1>
  <select id="runSelect"></select>
  <button id="gradeBtn" class="grade-btn" onclick="gradeTestSet()">Grade Test Set</button>
  <div class="status">
    <div class="dot" id="statusDot"></div>
    <span class="refresh-info" id="refreshInfo">Auto-refresh: 10s</span>
  </div>
</div>

<div class="nav-tabs">
  <div class="nav-tab active" data-page="overview">Overview</div>
  <div class="nav-tab" data-page="socrates">Socrates Reviews <span id="socratesCount" style="color:var(--purple)"></span></div>
  <div class="nav-tab" data-page="code">Code & Logs</div>
  <div class="nav-tab" data-page="step-detail" id="stepDetailTab" style="display:none">Step Detail</div>
</div>

<!-- PAGE: Overview -->
<div class="page active" id="page-overview">
<div class="grid">
  <div class="card full">
    <h2>Overview</h2>
    <div class="kpi-row">
      <div class="kpi"><div class="label">Steps</div><div class="value accent" id="kpiSteps">-</div></div>
      <div class="kpi"><div class="label">Best Train-Val Score</div><div class="value green" id="kpiBest">-</div></div>
      <div class="kpi"><div class="label">Best Test Score</div><div class="value orange" id="kpiTestScore">-</div></div>
      <div class="kpi"><div class="label">Good Nodes</div><div class="value green" id="kpiGood">-</div></div>
      <div class="kpi"><div class="label">Buggy Nodes</div><div class="value red" id="kpiBuggy">-</div></div>
      <div class="kpi"><div class="label">Pending</div><div class="value orange" id="kpiPending">-</div></div>
      <div class="kpi"><div class="label">Direction</div><div class="value" id="kpiDir">-</div></div>
    </div>
  </div>
  <div class="card">
    <h2>Validation Score</h2>
    <div class="chart-box"><canvas id="metricChart"></canvas></div>
  </div>
  <div class="card">
    <h2>Test Score</h2>
    <div class="chart-box"><canvas id="testChart"></canvas></div>
  </div>
  <!-- DISABLED: Stage Distribution
  <div class="card">
    <h2>Stage Distribution</h2>
    <div class="chart-box"><canvas id="stageChart"></canvas></div>
  </div>
  -->
  <div class="card full">
    <h2>Nodes (newest first)</h2>
    <div class="table-scroll">
      <table class="node-table">
        <thead><tr>
          <th>Step</th><th>ID</th><th>Stage</th><th>Train-Val</th><th>Test Score</th><th>Buggy</th><th>Exec Time</th><th>Time</th><th>Plan</th>
        </tr></thead>
        <tbody id="nodeTableBody"></tbody>
      </table>
    </div>
  </div>
</div>
</div>

<!-- PAGE: Step Detail -->
<div class="page" id="page-step-detail">
  <div class="step-detail">
    <div class="detail-header">
      <button class="back-btn" onclick="goBackToOverview()">Back</button>
      <h2 id="detailTitle">Step</h2>
    </div>
    <div class="detail-meta" id="detailMeta"></div>
    <div id="detailSections"></div>
  </div>
</div>

<!-- PAGE: Socrates Reviews -->
<div class="page" id="page-socrates">
  <div class="socrates-list" id="socratesList">
    <div class="no-socrates">No Socrates reviews yet. They appear once improve nodes trigger the review loop.</div>
  </div>
</div>

<!-- PAGE: Code & Logs -->
<div class="page" id="page-code">
<div class="grid">
  <div class="card">
    <h2>Best Solution Code</h2>
    <div class="code-wrapper">
      <button class="copy-btn" onclick="copyCode()">Copy</button>
      <pre class="code-block"><code class="language-python" id="bestCode">No solution yet...</code></pre>
    </div>
  </div>
  <div class="card">
    <h2>Recent Logs</h2>
    <pre class="log-block" id="logTail">Loading...</pre>
  </div>
</div>
</div>

<script>
let metricChart = null, testChart = null, stageChart = null;
let currentRun = null;
let lastRunData = null;
let refreshInterval = null;
let grading = false;
let activeTab = 'overview';
let codeLogsLoaded = false;
let socratesLoaded = false;

// Tab navigation
document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    activeTab = tab.dataset.page;
    document.getElementById('page-' + activeTab).classList.add('active');
    if (activeTab === 'code' && !codeLogsLoaded && currentRun) loadCodeLogs(currentRun);
    if (activeTab === 'socrates' && !socratesLoaded && currentRun) loadSocrates(currentRun);
  });
});

async function loadRuns() {
  const resp = await fetch('/api/runs');
  const runs = await resp.json();
  const sel = document.getElementById('runSelect');
  sel.innerHTML = '';
  runs.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r.name;
    opt.textContent = r.name;
    sel.appendChild(opt);
  });
  if (runs.length > 0 && !currentRun) {
    currentRun = runs[0].name;
  }
  sel.value = currentRun;
}

async function loadRun(runName) {
  const resp = await fetch('/api/run/' + runName);
  if (!resp.ok) return;
  const data = await resp.json();
  render(data);
}

function medalBadge(medal) {
  if (!medal) return '';
  const cls = medal === 'GOLD' ? 'gold' : medal === 'SILVER' ? 'silver' : medal === 'BRONZE' ? 'bronze' : medal === 'ABOVE MEDIAN' ? 'above-median' : 'below';
  return '<span class="medal-badge ' + cls + '">' + medal + '</span>';
}

function render(d) {
  lastRunData = d;
  // KPIs
  const totalCfg = d.total_steps_cfg || '?';
  document.getElementById('kpiSteps').textContent = d.total_nodes + ' / ' + totalCfg;
  const bp = d.best_progression;
  const bestVal = bp.length > 0 ? bp[bp.length - 1].best : null;
  document.getElementById('kpiBest').textContent = bestVal !== null ? bestVal.toFixed(6) : '-';
  document.getElementById('kpiGood').textContent = d.good;
  document.getElementById('kpiBuggy').textContent = d.buggy;
  document.getElementById('kpiPending').textContent = d.pending;
  document.getElementById('kpiDir').textContent = d.maximize === true ? 'Maximize' : d.maximize === false ? 'Minimize' : '?';

  // Test scores
  const ts = d.test_scores;
  const scores = ts ? ts.scores || {} : {};
  const kpiTest = document.getElementById('kpiTestScore');
  const btn = document.getElementById('gradeBtn');

  if (ts && Object.keys(scores).length > 0) {
    const isLower = ts.is_lower_better;
    let bestTestScore = null;
    let bestMedal = null;
    for (const [id, s] of Object.entries(scores)) {
      if (s.score != null) {
        if (bestTestScore == null || (isLower ? s.score < bestTestScore : s.score > bestTestScore)) {
          bestTestScore = s.score;
          bestMedal = s.medal;
        }
      }
    }
    if (bestTestScore != null) {
      kpiTest.innerHTML = bestTestScore.toFixed(6) + ' ' + medalBadge(bestMedal);
    } else {
      kpiTest.textContent = '-';
    }
    btn.textContent = 'Re-grade';
    if (ts.graded_at) btn.title = 'Last graded: ' + new Date(ts.graded_at).toLocaleString();
  } else {
    kpiTest.textContent = '-';
    btn.textContent = 'Grade Test Set';
    btn.title = '';
  }

  // Build node id -> step map and test score chart data
  const nodeStepMap = {};
  d.nodes.forEach(n => { nodeStepMap[n.id] = n.step; });
  const testData = [];
  for (const [id, s] of Object.entries(scores)) {
    if (s.score != null && nodeStepMap[id] != null) {
      testData.push({ step: nodeStepMap[id], score: s.score });
    }
  }
  testData.sort((a, b) => a.step - b.step);

  renderMetricChart(bp);
  renderTestChart(testData, ts ? ts.thresholds : null);
  renderStageChart(d.stages);

  // Node table
  const tbody = document.getElementById('nodeTableBody');
  tbody.innerHTML = '';
  const reversed = [...d.nodes].reverse().filter(n => n.stage !== 'root');
  reversed.forEach(n => {
    const tr = document.createElement('tr');
    const buggyClass = n.is_buggy === true ? 'buggy-true' : n.is_buggy === false ? 'buggy-false' : '';
    const buggyText = n.is_buggy === true ? 'YES' : n.is_buggy === false ? 'NO' : '...';
    const metricText = n.metric !== null ? n.metric.toFixed(6) : '-';
    const execText = n.exec_time !== null ? n.exec_time.toFixed(1) + 's' : '-';
    const rawTime = n.finish_time || n.created_time || '';
    let timeText = '-';
    if (rawTime) {
      const m = rawTime.match(/T(\d{2}):(\d{2})/);
      if (m) {
        let h = parseInt(m[1]), mm = m[2];
        const ampm = h >= 12 ? 'PM' : 'AM';
        h = h % 12 || 12;
        timeText = h + ':' + mm + ' ' + ampm;
      } else { timeText = rawTime; }
    }
    const nodeScore = scores[n.id];
    const testText = nodeScore && nodeScore.score != null ? nodeScore.score.toFixed(6) + ' ' + medalBadge(nodeScore.medal) : '-';
    tr.innerHTML = `
      <td>${n.step}</td>
      <td><code>${n.id}</code></td>
      <td><span class="badge ${n.stage}">${n.stage}</span></td>
      <td>${metricText}</td>
      <td>${testText}</td>
      <td class="${buggyClass}">${buggyText}</td>
      <td>${execText}</td>
      <td>${timeText}</td>
      <td><div class="plan-text">${escapeHtml(n.plan)}</div></td>
    `;
    tr.addEventListener('dblclick', () => {
      openStepDetail(n);
    });
    tbody.appendChild(tr);
  });

  // Lazy-load tab-specific data if that tab is active
  if (activeTab === 'code') loadCodeLogs(d.run_name);
  if (activeTab === 'socrates') loadSocrates(d.run_name);
}

async function loadCodeLogs(runName) {
  const resp = await fetch('/api/run/' + runName + '/code-logs');
  if (!resp.ok) return;
  const d = await resp.json();
  const codeEl = document.getElementById('bestCode');
  codeEl.textContent = d.best_code || 'No solution yet...';
  if (d.best_code) hljs.highlightElement(codeEl);
  document.getElementById('logTail').textContent = d.log_tail || 'No logs yet...';
  const logEl = document.getElementById('logTail');
  logEl.scrollTop = logEl.scrollHeight;
  codeLogsLoaded = true;
}

async function loadSocrates(runName) {
  const resp = await fetch('/api/run/' + runName + '/socrates');
  if (!resp.ok) return;
  const sessions = await resp.json();
  renderSocrates(sessions);
  socratesLoaded = true;
}

// Track which Socrates sessions the user has manually toggled
let socratesOpenState = {};  // keyed by session num, true = open
let socratesInitialized = false;

function renderSocrates(sessions) {
  const container = document.getElementById('socratesList');
  const countEl = document.getElementById('socratesCount');
  countEl.textContent = sessions.length > 0 ? `(${sessions.length})` : '';

  if (sessions.length === 0) {
    container.innerHTML = '<div class="no-socrates">No Socrates reviews yet. They appear once improve nodes trigger the review loop.</div>';
    socratesInitialized = false;
    return;
  }

  // On first render, default newest open
  if (!socratesInitialized) {
    socratesOpenState = {};
    socratesOpenState[sessions.length] = true;
    socratesInitialized = true;
  }

  container.innerHTML = '';
  // Show newest first
  sessions.slice().reverse().forEach((session, idx) => {
    const num = sessions.length - idx;
    const div = document.createElement('div');
    const isOpen = socratesOpenState[num] === true;
    div.className = 'socrates-session' + (isOpen ? ' open' : '');

    const approvedClass = session.approved ? 'yes' : 'no';
    const approvedText = session.approved ? 'APPROVED' : 'NOT APPROVED';

    let headerHtml = `
      <div class="socrates-session-header">
        <span class="session-num">#${num}</span>
        <span class="session-time">${session.timestamp || ''}</span>
        <span class="session-plan">${escapeHtml(session.original_plan)}</span>
        <span class="approved-badge ${approvedClass}">${approvedText}</span>
        <span class="rounds-info">${session.rounds} round${session.rounds !== 1 ? 's' : ''}</span>
        <span class="chevron">&#9654;</span>
      </div>
    `;

    let bodyHtml = '<div class="socrates-session-body">';
    (session.transcript || []).forEach(round => {
      bodyHtml += `<div class="socrates-round">`;
      bodyHtml += `<div class="round-label">Round ${round.round}</div>`;
      bodyHtml += `<div class="socrates-msg socrates"><div class="msg-role socrates">Socrates (PI)</div>${escapeHtml(round.socrates)}</div>`;
      if (round.planner) {
        bodyHtml += `<div class="socrates-msg planner"><div class="msg-role planner">Planner</div>${escapeHtml(round.planner)}</div>`;
      }
      if (round.approved) {
        bodyHtml += `<div style="padding:8px 16px;color:var(--green);font-size:11px;font-weight:600;">Plan approved this round</div>`;
      }
      bodyHtml += `</div>`;
    });
    bodyHtml += '</div>';

    div.innerHTML = headerHtml + bodyHtml;

    // Toggle open/close — persist to state so refresh preserves it
    div.querySelector('.socrates-session-header').addEventListener('click', () => {
      const nowOpen = !div.classList.contains('open');
      div.classList.toggle('open');
      socratesOpenState[num] = nowOpen;
    });

    container.appendChild(div);
  });
}

function renderMetricChart(bp) {
  const ctx = document.getElementById('metricChart').getContext('2d');
  if (metricChart) metricChart.destroy();

  const filtered = bp.filter(p => (p.value === null || p.value <= 4) && (p.best === null || p.best <= 4));

  const datasets = [
    { label: 'Best', data: filtered.map(p => p.best), borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.1)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 },
    { label: 'Per-node', data: filtered.map(p => p.value), borderColor: 'rgba(88,166,255,0.4)', borderWidth: 1, pointRadius: 2, pointBackgroundColor: '#58a6ff', fill: false }
  ];

  metricChart = new Chart(ctx, {
    type: 'line',
    data: { labels: filtered.map(p => p.step), datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#8b949e', font: { size: 11 } } } },
      scales: {
        x: { title: { display: true, text: 'Step', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.5)' } },
        y: { title: { display: true, text: 'Score', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.5)' }, max: 4 }
      }
    }
  });
}

function renderTestChart(testData, thresholds) {
  const ctx = document.getElementById('testChart').getContext('2d');
  if (testChart) testChart.destroy();

  const annotations = {};
  if (thresholds) {
    const defs = [
      { key: 'gold', color: '#FFD700', label: 'Gold' },
      { key: 'silver', color: '#C0C0C0', label: 'Silver' },
      { key: 'bronze', color: '#CD7F32', label: 'Bronze' },
      { key: 'median', color: '#3fb950', label: 'Median' }
    ];
    defs.forEach(t => {
      if (thresholds[t.key] != null) {
        annotations[t.key + 'Line'] = {
          type: 'line', yMin: thresholds[t.key], yMax: thresholds[t.key],
          borderColor: t.color, borderWidth: 1, borderDash: [6, 3],
          label: { display: true, content: t.label, position: 'end', color: t.color, font: { size: 9 }, backgroundColor: 'rgba(13,17,23,0.8)', padding: 2 }
        };
      }
    });
  }

  const datasets = [];
  const filteredTestData = testData ? testData.filter(t => t.score <= 4) : [];
  if (filteredTestData.length > 0) {
    datasets.push({
      label: 'Test Score', data: filteredTestData.map(t => ({ x: t.step, y: t.score })),
      borderColor: '#d29922', backgroundColor: 'rgba(210,153,34,0.1)', fill: true,
      borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#d29922'
    });
  }

  testChart = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#8b949e', font: { size: 11 } } },
        annotation: { annotations }
      },
      scales: {
        x: { type: 'linear', title: { display: true, text: 'Step', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.5)' } },
        y: { title: { display: true, text: 'Score', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.5)' }, max: 4 }
      }
    }
  });
}

/* DISABLED: Stage Distribution chart
function renderStageChart(stages) {
  const ctx = document.getElementById('stageChart').getContext('2d');
  if (stageChart) stageChart.destroy();
  const labels = Object.keys(stages);
  const values = Object.values(stages);
  const colors = { draft: '#58a6ff', improve: '#3fb950', debug: '#f85149', evolution: '#bc8cff', fusion: '#d29922', fusion_draft: '#d29922' };
  stageChart = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data: values, backgroundColor: labels.map(l => colors[l] || '#8b949e'), borderColor: '#161b22', borderWidth: 2 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { color: '#e6edf3', font: { size: 11 }, padding: 8 } } } }
  });
}
*/
function renderStageChart(stages) {}

async function openStepDetail(nodeStub) {
  // Show loading state immediately
  const tab = document.getElementById('stepDetailTab');
  tab.style.display = '';
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  tab.classList.add('active');
  activeTab = 'step-detail';
  document.getElementById('page-step-detail').classList.add('active');
  document.getElementById('detailTitle').textContent = 'Step ' + nodeStub.step + ' — ' + nodeStub.id;
  document.getElementById('detailSections').innerHTML = '<div style="color:var(--text2);padding:40px;text-align:center">Loading...</div>';

  // Fetch full detail from server
  const resp = await fetch('/api/run/' + currentRun + '/node/' + nodeStub.id);
  if (!resp.ok) {
    document.getElementById('detailSections').innerHTML = '<div style="color:var(--red);padding:40px">Failed to load node detail</div>';
    return;
  }
  const node = await resp.json();
  const parseMd = typeof marked.parse === 'function' ? marked.parse : marked;

  const metricText = node.metric !== null ? node.metric.toFixed(6) : '-';
  const execText = node.exec_time !== null ? node.exec_time.toFixed(1) + 's' : '-';
  document.getElementById('detailMeta').innerHTML =
    '<span>Stage: <span class="badge ' + node.stage + '">' + node.stage + '</span></span>' +
    '<span>Score: ' + metricText + '</span>' +
    '<span>Buggy: ' + (node.is_buggy === true ? 'YES' : node.is_buggy === false ? 'NO' : '...') + '</span>' +
    '<span>Exec: ' + execText + '</span>';

  let html = '';
  if (node.plan) {
    html += '<div class="card" style="margin-bottom:12px"><h2>Plan</h2><div class="plan-content">' + parseMd(node.plan) + '</div></div>';
  }
  if (node.analysis) {
    html += '<div class="card" style="margin-bottom:12px"><h2>Analysis</h2><div class="plan-content">' + parseMd(node.analysis) + '</div></div>';
  }
  if (node.code_summary) {
    html += '<div class="card" style="margin-bottom:12px"><h2>Code Summary</h2><div class="plan-content">' + parseMd(node.code_summary) + '</div></div>';
  }
  const termOut = (node.term_out || []).join('');
  if (termOut) {
    html += '<div class="card" style="margin-bottom:12px"><h2>Execution Output</h2><pre class="log-block" style="max-height:400px">' + escapeHtml(termOut) + '</pre></div>';
  }
  if (node.exc_type) {
    const errMsg = node.exc_info ? (node.exc_info.message || JSON.stringify(node.exc_info)) : '';
    html += '<div class="card" style="margin-bottom:12px"><h2>Error</h2><pre class="log-block" style="color:var(--red)">' + escapeHtml(node.exc_type + ': ' + errMsg) + '</pre></div>';
  }
  if (node.code) {
    html += '<div class="card" style="margin-bottom:12px"><h2>Code</h2><div class="code-wrapper"><button class="copy-btn" onclick="navigator.clipboard.writeText(document.getElementById(\'detailCode\').textContent).then(()=>{this.textContent=\'Copied!\';setTimeout(()=>this.textContent=\'Copy\',1500)})">Copy</button><pre class="code-block"><code class="language-python" id="detailCode">' + escapeHtml(node.code) + '</code></pre></div></div>';
  }
  document.getElementById('detailSections').innerHTML = html;
  const codeBlock = document.getElementById('detailCode');
  if (codeBlock) hljs.highlightElement(codeBlock);
}

function goBackToOverview() {
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelector('[data-page="overview"]').classList.add('active');
  document.getElementById('page-overview').classList.add('active');
}

function copyCode() {
  const text = document.getElementById('bestCode').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('.copy-btn');
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 1500);
  });
}

function escapeHtml(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function gradeTestSet() {
  if (!currentRun || grading) return;
  grading = true;
  const btn = document.getElementById('gradeBtn');
  btn.textContent = 'Grading...';
  btn.disabled = true;
  const resp = await fetch('/api/run/' + currentRun + '/grade', { method: 'POST' });
  grading = false;
  btn.disabled = false;
  if (resp.ok) {
    await loadRun(currentRun);
  } else {
    const data = await resp.json();
    btn.textContent = 'Grade Test Set';
    alert('Grading failed: ' + (data.error || 'unknown error'));
  }
}

document.getElementById('runSelect').addEventListener('change', e => {
  currentRun = e.target.value;
  socratesOpenState = {};
  socratesInitialized = false;
  codeLogsLoaded = false;
  socratesLoaded = false;
  loadRun(currentRun);
});

async function refresh() {
  await loadRuns();
  codeLogsLoaded = false;
  socratesLoaded = false;
  if (currentRun) await loadRun(currentRun);
  document.getElementById('refreshInfo').textContent = 'Updated: ' + new Date().toLocaleTimeString() + ' (10s)';
}

refresh();
refreshInterval = setInterval(refresh, 10000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    print(f"Dashboard: http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
