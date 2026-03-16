"""Export dashboard API responses as static JSON + index.html for Vercel deployment."""
import sys
import json
import argparse
from pathlib import Path

# Re-use data loaders from the live dashboard
sys.path.insert(0, str(Path(__file__).parent))
from dashboard import (
    find_runs, load_journal, load_best_code, load_config,
    load_log_tail, load_socrates_transcripts, load_test_scores,
)

TERM_OUT_CAP = 100_000  # 100KB cap per node


def cap_term_out(lines):
    """Keep last 50 lines, cap total at 100KB."""
    if not lines:
        return lines
    joined = "".join(lines)
    if len(joined) <= TERM_OUT_CAP:
        return lines
    tail = lines[-50:]
    joined = "".join(tail)
    if len(joined) > TERM_OUT_CAP:
        joined = joined[-TERM_OUT_CAP:]
        tail = [joined]
    return ["... [truncated — showing last portion] ...\n"] + tail


def build_summary(run_name):
    """Replicate the /api/run/<name> endpoint logic."""
    journal = load_journal(run_name)
    if journal is None:
        return None

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

    return {
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
    }


def build_node_detail(node):
    """Replicate /api/run/<name>/node/<id> endpoint logic."""
    return {
        "step": node["step"],
        "id": node["id"][:8],
        "stage": node["stage"],
        "metric": node["metric"]["value"],
        "is_buggy": node["is_buggy"],
        "exec_time": node.get("exec_time"),
        "plan": node.get("plan") or "",
        "analysis": node.get("analysis") or "",
        "code_summary": node.get("code_summary") or "",
        "term_out": cap_term_out(node.get("_term_out") or []),
        "exc_type": node.get("exc_type"),
        "exc_info": node.get("exc_info"),
        "code": node.get("code") or "",
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


def export(output_dir):
    out = Path(output_dir)
    runs = find_runs()
    print(f"Found {len(runs)} runs")

    # runs.json
    write_json(out / "api" / "runs.json", runs)

    for run in runs:
        name = run["name"]
        print(f"\n--- {name} ---")
        run_dir = out / "api" / "run" / name

        # summary.json
        print("  Loading journal...")
        summary = build_summary(name)
        if summary is None:
            print("  SKIP (no journal)")
            continue
        write_json(run_dir / "summary.json", summary)
        print(f"  summary: {len(summary['nodes'])} nodes")

        # per-node detail JSONs
        journal = load_journal(name)
        nodes_dir = run_dir / "nodes"
        node_count = 0
        for n in journal["nodes"]:
            short_id = n["id"][:8]
            detail = build_node_detail(n)
            write_json(nodes_dir / f"{short_id}.json", detail)
            node_count += 1
        print(f"  nodes: {node_count} files")

        # code-logs.json
        write_json(run_dir / "code-logs.json", {
            "best_code": load_best_code(name),
            "log_tail": load_log_tail(name),
        })
        print("  code-logs: done")

        # socrates.json
        transcripts = load_socrates_transcripts(name)
        write_json(run_dir / "socrates.json", transcripts)
        print(f"  socrates: {len(transcripts)} sessions")

    # Copy static index.html
    html_src = Path(__file__).parent / "static_dashboard.html"
    html_dst = out / "index.html"
    html_dst.write_text(html_src.read_text())
    print(f"\nindex.html copied")

    # Report total size
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nTotal output: {total / 1024 / 1024:.1f} MB in {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dist")
    args = parser.parse_args()
    export(args.output_dir)
