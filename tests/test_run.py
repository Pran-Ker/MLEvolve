#!/usr/bin/env python3
"""
Standalone test runner: execute a solution file and report validation + test scores with full logs.

Usage:
    # Interactive picker, both validation and test runs
    python tests/test_run.py

    # Direct file
    python tests/test_run.py my_solution.py

    # Validation only / test only
    python tests/test_run.py --val-only
    python tests/test_run.py --test-only

    # Custom timeout
    python tests/test_run.py --timeout 600
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(__file__).resolve().parent / "runs"
sys.path.insert(0, str(PROJECT_ROOT))

# Defaults
DEFAULT_DATA_DIR = "/Users/pran-ker/Library/Caches/mle-bench/data/ventilator-pressure-prediction/prepared/public"
DEFAULT_EXP_ID = "ventilator-pressure-prediction"
VAL_SPLIT = 0.15
VAL_SEED = 42
TARGET_COL = "pressure"
ID_COL = "id"
GROUP_COL = "breath_id"


# ── helpers ──────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


def print_banner(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def pick_solution():
    py_files = sorted(p for p in RUNS_DIR.glob("*.py") if p.name != "__init__.py")
    if not py_files:
        print(f"No Python files found in {RUNS_DIR}")
        sys.exit(1)

    print(f"\nAvailable solutions in {RUNS_DIR}:\n")
    for i, p in enumerate(py_files, 1):
        print(f"  [{i}] {p.name}")
    print()

    while True:
        choice = input("Pick a file number (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(py_files):
            return py_files[int(choice) - 1]
        print(f"  Invalid choice. Enter 1-{len(py_files)}.")


# ── workspace setup ──────────────────────────────────────────────────────

def make_workspace(data_dir):
    """Create a fresh temp workspace with data symlinked into input/."""
    ws = Path(tempfile.mkdtemp(prefix="mlevolve_test_"))
    for sub in ("input", "working", "submission"):
        (ws / sub).mkdir()
    for item in Path(data_dir).iterdir():
        dst = ws / "input" / item.name
        if not dst.exists():
            dst.symlink_to(item)
    return ws


def make_val_workspace(data_dir):
    """Create workspace with an 85/15 breath-level split of train.csv.

    - input/train.csv  → 85% of breaths
    - input/test.csv   → 15% of breaths (target removed)
    - input/sample_submission.csv → matching ids with dummy target
    - ground truth saved separately for scoring
    Returns (workspace_path, ground_truth_df).
    """
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")

    breath_ids = train_df[GROUP_COL].unique()
    rng = np.random.RandomState(VAL_SEED)
    rng.shuffle(breath_ids)
    split_idx = int(len(breath_ids) * (1 - VAL_SPLIT))
    train_breaths = set(breath_ids[:split_idx])
    val_breaths = set(breath_ids[split_idx:])

    train_split = train_df[train_df[GROUP_COL].isin(train_breaths)]
    val_split = train_df[train_df[GROUP_COL].isin(val_breaths)]

    # Ground truth for scoring
    ground_truth = val_split[[ID_COL, TARGET_COL, "u_out"]].copy()

    # Build test csv (no target)
    test_cols = [c for c in val_split.columns if c != TARGET_COL]
    val_test = val_split[test_cols]

    # Build sample submission
    sample_sub = val_split[[ID_COL]].copy()
    sample_sub[TARGET_COL] = 0

    # Create workspace
    ws = Path(tempfile.mkdtemp(prefix="mlevolve_val_"))
    for sub in ("input", "working", "submission"):
        (ws / sub).mkdir()

    train_split.to_csv(ws / "input" / "train.csv", index=False)
    val_test.to_csv(ws / "input" / "test.csv", index=False)
    sample_sub.to_csv(ws / "input" / "sample_submission.csv", index=False)

    # Copy any other files from data_dir (description.md, etc.)
    for item in data_dir.iterdir():
        if item.name in ("train.csv", "test.csv", "sample_submission.csv"):
            continue
        dst = ws / "input" / item.name
        if not dst.exists():
            dst.symlink_to(item)

    return ws, ground_truth


# ── execution ────────────────────────────────────────────────────────────

def run_solution(solution_path, workspace_dir, timeout):
    solution_path = Path(solution_path).resolve()
    workspace_dir = Path(workspace_dir).resolve()

    if not solution_path.exists():
        return {"error": f"Solution file not found: {solution_path}"}

    runfile = workspace_dir / "runfile_test.py"
    shutil.copy2(solution_path, runfile)

    start = time.time()
    timed_out = False

    proc = subprocess.Popen(
        [sys.executable, str(runfile)],
        cwd=str(workspace_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.send_signal(signal.SIGINT)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

    elapsed = time.time() - start
    runfile.unlink(missing_ok=True)

    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": proc.returncode,
        "elapsed": elapsed,
        "timed_out": timed_out,
    }


def find_submission(workspace_dir):
    workspace_dir = Path(workspace_dir)
    candidates = [
        workspace_dir / "submission" / "submission.csv",
        workspace_dir / "submission.csv",
    ]
    for p in sorted(workspace_dir.rglob("submission*.csv")):
        if p not in candidates:
            candidates.append(p)
    for p in candidates:
        if p.exists():
            return p
    return None


# ── scoring ──────────────────────────────────────────────────────────────

def score_validation(submission_path, ground_truth):
    """Compute MAE on inspiratory phase (u_out==0) matching the competition metric."""
    sub_df = pd.read_csv(submission_path)
    merged = ground_truth.merge(sub_df, on=ID_COL, suffixes=("_true", "_pred"))

    # Overall MAE
    mae_all = np.abs(merged[f"{TARGET_COL}_true"] - merged[f"{TARGET_COL}_pred"]).mean()

    # Inspiratory-only MAE (competition metric: only u_out==0 is scored)
    insp = merged[merged["u_out"] == 0]
    mae_insp = np.abs(insp[f"{TARGET_COL}_true"] - insp[f"{TARGET_COL}_pred"]).mean()

    return {
        "mae_inspiratory": mae_insp,
        "mae_all": mae_all,
        "n_rows": len(merged),
        "n_inspiratory": len(insp),
    }


def validate_format(submission_path, exp_id):
    from engine.validation.format_client import call_validate, is_server_online

    online, url = is_server_online(max_retries=1, timeout=5)
    if not online:
        return None, "Grading server not running (skipped)"
    status, res = call_validate(exp_id=exp_id, submission_path=submission_path)
    return status, res


def validate_content_quality(submission_path):
    from engine.validation.quality_check import validate_submission_content_quality

    return validate_submission_content_quality(
        submission_path=submission_path,
        sample_path=None,
        constant_threshold=0.95,
    )


# ── run phase (shared logic) ────────────────────────────────────────────

def run_phase(label, solution_path, workspace_dir, timeout):
    """Execute a solution and print all logs. Returns (result_dict, submission_path)."""
    print_banner(f"{label}: {solution_path.name}")
    print(f"  Workspace: {workspace_dir}")
    print(f"  Timeout:   {timeout}s")

    result = run_solution(solution_path, workspace_dir, timeout)

    if "error" in result:
        print(f"  Error: {result['error']}")
        return result, None

    # Logs
    print_section(f"{label} STDOUT")
    print(result["stdout"] if result["stdout"].strip() else "  (empty)")

    if result["stderr"].strip():
        print_section(f"{label} STDERR")
        print(result["stderr"])

    # Execution summary
    print_section(f"{label} EXECUTION")
    status_str = "TIMEOUT" if result["timed_out"] else ("OK" if result["returncode"] == 0 else "FAILED")
    print(f"  Status:      {status_str}")
    print(f"  Return code: {result['returncode']}")
    print(f"  Time:        {result['elapsed']:.1f}s")

    # Submission check
    submission_path = find_submission(workspace_dir)
    print_section(f"{label} SUBMISSION")
    if submission_path:
        df = pd.read_csv(submission_path)
        print(f"  Found:   {submission_path}")
        print(f"  Shape:   {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Head:\n{df.head().to_string(index=False, max_cols=8)}")

        is_valid, err = validate_content_quality(submission_path)
        print(f"  Quality: {'PASSED' if is_valid else f'FAILED - {err}'}")
    else:
        print("  No submission file found")

    return result, submission_path


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run and validate an MLEvolve solution file")
    parser.add_argument("solution", nargs="?", default=None, help="Path to solution (omit to pick from tests/runs/)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to task data")
    parser.add_argument("--exp-id", default=DEFAULT_EXP_ID, help="Experiment ID for grading server")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per run in seconds (default: 3600)")
    parser.add_argument("--val-only", action="store_true", help="Only run validation (85/15 split)")
    parser.add_argument("--test-only", action="store_true", help="Only run test (full data)")
    parser.add_argument("--keep-workspace", action="store_true", help="Keep temp workspaces after run")
    args = parser.parse_args()

    solution_path = Path(args.solution).resolve() if args.solution else pick_solution()
    if not solution_path.exists():
        print(f"Error: solution file not found: {solution_path}")
        sys.exit(1)

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"Error: data dir not found: {data_dir}")
        sys.exit(1)

    run_val = not args.test_only
    run_test = not args.val_only

    val_score = None
    test_score = None
    workspaces = []

    # ── Validation run (85/15 split) ──
    if run_val:
        val_ws, ground_truth = make_val_workspace(data_dir)
        workspaces.append(val_ws)
        n_train = len(pd.read_csv(val_ws / "input" / "train.csv"))
        n_test = len(ground_truth)
        print(f"\n  Split: {n_train} train rows, {n_test} val rows ({VAL_SPLIT*100:.0f}%), seed={VAL_SEED}")

        result, sub_path = run_phase("VALIDATION", solution_path, val_ws, args.timeout)

        if sub_path:
            val_score = score_validation(sub_path, ground_truth)

    # ── Test run (full data) ──
    if run_test:
        test_ws = make_workspace(data_dir)
        workspaces.append(test_ws)

        result, sub_path = run_phase("TEST", solution_path, test_ws, args.timeout)

        if sub_path and args.exp_id:
            print_section("TEST FORMAT VALIDATION (grading server)")
            status, res = validate_format(sub_path, args.exp_id)
            if status is None:
                print(f"  {res}")
            elif status and isinstance(res, dict):
                if res.get("is_valid"):
                    print("  PASSED")
                    if "score" in res:
                        test_score = res["score"]
                else:
                    print(f"  FAILED: {res.get('result', res)}")
            else:
                print(f"  ERROR: {res}")

    # ── Final scoreboard ──
    print_banner("SCORES")

    if val_score:
        print(f"  Validation MAE (inspiratory): {val_score['mae_inspiratory']:.6f}")
        print(f"  Validation MAE (all rows):    {val_score['mae_all']:.6f}")
        print(f"  Scored rows: {val_score['n_inspiratory']} inspiratory / {val_score['n_rows']} total")
    elif run_val:
        print("  Validation: no score (submission missing or execution failed)")

    if test_score is not None:
        print(f"  Test score (grading server):  {test_score}")
    elif run_test:
        print("  Test: grading server score not available")

    print()

    # Cleanup
    for ws in workspaces:
        if args.keep_workspace:
            print(f"  Workspace kept: {ws}")
        else:
            shutil.rmtree(ws, ignore_errors=True)

    if not args.keep_workspace and workspaces:
        print("  Cleaned up temp workspaces")


if __name__ == "__main__":
    main()
