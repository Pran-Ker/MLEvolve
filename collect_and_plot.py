"""
Read journal files from statoil Socrates vs Baseline runs and generate comparison plots.
Output: experiments/statoil_comparison.png
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("experiments")
OUTPUT_FILE = OUTPUT_DIR / "statoil_comparison.png"

# Match patterns for finding the right run directories
PATTERNS = {
    "Socrates": "statoil_socrates_seed42",
    "Baseline": "statoil_baseline_seed42",
}

COLORS = {"Socrates": "#2196F3", "Baseline": "#FF9800"}


def find_journal(pattern: str) -> Path | None:
    """Find the most recent journal.json matching the exp_name pattern."""
    candidates = sorted(RUNS_DIR.glob(f"*{pattern}/logs/journal.json"), reverse=True)
    return candidates[0] if candidates else None


def load_journal(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_metrics(journal: dict) -> list[dict]:
    """Extract per-node info: step, stage, metric value, ctime, is_buggy, maximize."""
    nodes = journal["nodes"]
    maximize = None
    for n in nodes:
        if n["metric"]["maximize"] is not None:
            maximize = n["metric"]["maximize"]
            break

    results = []
    for n in nodes:
        results.append({
            "step": n["step"],
            "stage": n["stage"],
            "metric": n["metric"]["value"],
            "is_buggy": n["is_buggy"],
            "ctime": n["ctime"],
            "maximize": maximize,
        })
    return results


def best_metric_over_steps(nodes: list[dict]) -> tuple[list[int], list[float]]:
    """Compute running best metric over steps (handles maximize/minimize)."""
    maximize = nodes[0]["maximize"]
    steps, bests = [], []
    current_best = None
    for n in sorted(nodes, key=lambda x: x["step"]):
        v = n["metric"]
        if v is None:
            continue
        if current_best is None:
            current_best = v
        elif maximize and v > current_best:
            current_best = v
        elif not maximize and v < current_best:
            current_best = v
        steps.append(n["step"])
        bests.append(current_best)
    return steps, bests


def time_to_best(nodes: list[dict]) -> tuple[float, float, int]:
    """Return (time_seconds, best_value, best_step) for when the best metric was first achieved."""
    maximize = nodes[0]["maximize"]
    valid = [n for n in nodes if n["metric"] is not None]
    if not valid:
        return 0.0, 0.0, 0

    if maximize:
        best_node = max(valid, key=lambda n: n["metric"])
    else:
        best_node = min(valid, key=lambda n: n["metric"])

    t0 = min(n["ctime"] for n in nodes)
    elapsed = best_node["ctime"] - t0
    return elapsed, best_node["metric"], best_node["step"]


def stage_distribution(nodes: list[dict]) -> Counter:
    return Counter(n["stage"] for n in nodes if n["stage"] != "root")


def plot_comparison(data: dict[str, list[dict]]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Statoil Iceberg: Socrates vs Baseline", fontsize=16, fontweight="bold")

    # 1) Best metric over steps
    ax = axes[0, 0]
    for label, nodes in data.items():
        steps, bests = best_metric_over_steps(nodes)
        ax.plot(steps, bests, label=label, color=COLORS[label], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Best Metric")
    ax.set_title("Best Metric Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Final best metric bar chart
    ax = axes[0, 1]
    labels, values = [], []
    for label, nodes in data.items():
        _, bests = best_metric_over_steps(nodes)
        labels.append(label)
        values.append(bests[-1] if bests else 0)
    bars = ax.bar(labels, values, color=[COLORS[l] for l in labels], width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.6f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Best Metric")
    ax.set_title("Final Best Metric")
    ax.grid(True, alpha=0.3, axis="y")

    # 3) Node stage distribution
    ax = axes[1, 0]
    stage_order = ["draft", "improve", "debug", "evolution", "fusion", "fusion_draft"]
    x_pos = range(len(stage_order))
    width = 0.35
    for i, (label, nodes) in enumerate(data.items()):
        dist = stage_distribution(nodes)
        counts = [dist.get(s, 0) for s in stage_order]
        offset = -width / 2 + i * width
        ax.bar([p + offset for p in x_pos], counts, width, label=label, color=COLORS[label])
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(stage_order, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Node Stage Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 4) Time to best
    ax = axes[1, 1]
    ttb_labels, ttb_times, ttb_steps = [], [], []
    for label, nodes in data.items():
        elapsed, best_val, best_step = time_to_best(nodes)
        ttb_labels.append(label)
        ttb_times.append(elapsed / 60)  # minutes
        ttb_steps.append(best_step)
    bars = ax.bar(ttb_labels, ttb_times, color=[COLORS[l] for l in ttb_labels], width=0.5)
    for bar, t, s in zip(bars, ttb_times, ttb_steps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.1f}min\n(step {s})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Minutes")
    ax.set_title("Time to Best Solution")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_FILE}")


def main():
    data = {}
    for label, pattern in PATTERNS.items():
        journal_path = find_journal(pattern)
        if journal_path is None:
            print(f"WARNING: No journal found for '{pattern}'. Skipping {label}.")
            continue
        print(f"Found {label}: {journal_path}")
        journal = load_journal(journal_path)
        data[label] = extract_metrics(journal)
        print(f"  Nodes: {len(data[label])}")

    if len(data) < 2:
        print("Need both Socrates and Baseline journals to generate comparison.")
        sys.exit(1)

    plot_comparison(data)


if __name__ == "__main__":
    main()
