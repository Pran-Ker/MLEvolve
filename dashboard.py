"""
Live MLEvolve experiment dashboard.
Usage: .venv/bin/python dashboard.py [--port 8050]
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)
RUNS_DIR = Path("runs")


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
    if not journal_path.exists():
        return None
    with open(journal_path) as f:
        return json.load(f)


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

    best_code = load_best_code(run_name)
    log_tail = load_log_tail(run_name)

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

    socrates = load_socrates_transcripts(run_name)
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
        "best_code": best_code,
        "log_tail": log_tail,
        "socrates": socrates,
        "test_scores": test_scores,
    })


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

  pre.code-block { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-size: 11px; max-height: 400px; overflow: auto; white-space: pre-wrap; word-break: break-all; line-height: 1.5; }
  pre.log-block { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-size: 11px; max-height: 300px; overflow: auto; white-space: pre-wrap; word-break: break-all; color: var(--text2); line-height: 1.4; }

  .plan-text { max-width: 400px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: pointer; }
  .plan-text:hover { white-space: normal; overflow: visible; }

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
    <h2>Score Progression</h2>
    <div class="chart-box"><canvas id="metricChart"></canvas></div>
  </div>
  <div class="card">
    <h2>Stage Distribution</h2>
    <div class="chart-box"><canvas id="stageChart"></canvas></div>
  </div>
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
    <pre class="code-block" id="bestCode">No solution yet...</pre>
  </div>
  <div class="card">
    <h2>Recent Logs</h2>
    <pre class="log-block" id="logTail">Loading...</pre>
  </div>
</div>
</div>

<script>
let metricChart = null, stageChart = null;
let currentRun = null;
let refreshInterval = null;
let grading = false;

// Tab navigation
document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('page-' + tab.dataset.page).classList.add('active');
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

  renderMetricChart(bp, testData, ts ? ts.thresholds : null);
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
    const timeText = n.finish_time || n.created_time || '-';
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
    tbody.appendChild(tr);
  });

  // Best code
  document.getElementById('bestCode').textContent = d.best_code || 'No solution yet...';

  // Logs
  document.getElementById('logTail').textContent = d.log_tail || 'No logs yet...';
  const logEl = document.getElementById('logTail');
  logEl.scrollTop = logEl.scrollHeight;

  // Socrates
  renderSocrates(d.socrates || []);
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

function renderMetricChart(bp, testData, thresholds) {
  const ctx = document.getElementById('metricChart').getContext('2d');
  if (metricChart) metricChart.destroy();

  const datasets = [
    { label: 'Best', data: bp.map(p => p.best), borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.1)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 },
    { label: 'Per-node', data: bp.map(p => p.value), borderColor: 'rgba(88,166,255,0.4)', borderWidth: 1, pointRadius: 2, pointBackgroundColor: '#58a6ff', fill: false }
  ];

  // Add test score overlay
  if (testData && testData.length > 0) {
    const testMap = {};
    testData.forEach(t => { testMap[t.step] = t.score; });
    const testValues = bp.map(p => testMap[p.step] !== undefined ? testMap[p.step] : null);
    datasets.push({
      label: 'Test Score', data: testValues, borderColor: '#d29922',
      borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#d29922', fill: false, spanGaps: false
    });
  }

  // Medal threshold annotations
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

  metricChart = new Chart(ctx, {
    type: 'line',
    data: { labels: bp.map(p => p.step), datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#8b949e', font: { size: 11 } } },
        annotation: { annotations }
      },
      scales: {
        x: { title: { display: true, text: 'Step', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.5)' } },
        y: { title: { display: true, text: 'Score', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: 'rgba(48,54,61,0.5)' } }
      }
    }
  });
}

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
  loadRun(currentRun);
});

async function refresh() {
  await loadRuns();
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
