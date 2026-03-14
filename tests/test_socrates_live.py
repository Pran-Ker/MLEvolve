#!/usr/bin/env python3
"""
Live Socrates review tester — runs real API calls, writes dashboard-compatible output.

Usage:
    python tests/test_socrates_live.py                          # default: improve agent
    python tests/test_socrates_live.py --stage evolution
    python tests/test_socrates_live.py --stage fusion
    python tests/test_socrates_live.py --stage draft
    python tests/test_socrates_live.py --stage improve --rounds 5
    python tests/test_socrates_live.py --run test               # creates a new test run
    python tests/test_socrates_live.py --run 20260308_163144_ventilator-pressure-prediction
    python tests/test_socrates_live.py --list-runs

Output:
    runs/<run>/logs/socrates_transcripts.jsonl   (viewable on dashboard)
    tests/socrates_live_{stage}.log              (human-readable)

Dashboard:
    python dashboard.py
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace
from unittest.mock import MagicMock


# ── Scenario definitions per stage ──

SCENARIOS = {
    "improve": {
        "plan": (
            "Replace the current logistic regression with LightGBM. "
            "Use target encoding for all 8 categorical features. "
            "Apply 5-fold cross-validation. "
            "Use early stopping with patience=30 on validation AUC."
        ),
        "agent_context": (
            "You are a Kaggle Grandmaster improving a customer churn prediction solution. "
            "The current approach uses logistic regression with one-hot encoding (AUC: 0.72). "
            "Your task is to propose and defend a better approach."
        ),
        "parent_output": "AUC: 0.72 | Precision: 0.65 | Recall: 0.58",
        "child_memory": "Attempt 1: LogisticRegression + OneHot, AUC=0.72",
        "task_desc": (
            "Predict customer churn (binary classification). "
            "Dataset: 1000 rows, 15 features (8 categorical, 7 numeric). "
            "Metric: AUC-ROC (higher is better). "
            "Class distribution: 80% no-churn, 20% churn."
        ),
    },
    "evolution": {
        "plan": (
            "Based on the evolution trajectory showing diminishing returns from tree-based methods, "
            "shift to a neural network approach. Use a 3-layer MLP with batch normalization "
            "and dropout=0.3. Implement learning rate warmup over 10 epochs then cosine decay. "
            "Use the same StratifiedKFold(5) split as previous iterations."
        ),
        "agent_context": (
            "You are a Kaggle Grandmaster evolving a house price prediction solution. "
            "The branch trajectory shows: Step 1 (RF, RMSE=0.42) -> Step 2 (XGBoost, RMSE=0.38) -> "
            "Step 3 (LightGBM + feature eng, RMSE=0.36) -> Step 4 (CatBoost tuned, RMSE=0.355). "
            "Tree methods are plateauing. You are proposing a paradigm shift to neural networks."
        ),
        "parent_output": "RMSE: 0.355 | R2: 0.87 | MAE: 0.28",
        "child_memory": (
            "Attempt 1: RF baseline, RMSE=0.42\n"
            "Attempt 2: XGBoost + basic features, RMSE=0.38\n"
            "Attempt 3: LightGBM + lag features, RMSE=0.36\n"
            "Attempt 4: CatBoost + tuning, RMSE=0.355"
        ),
        "task_desc": (
            "Predict house prices (regression). "
            "Dataset: 5000 rows, 25 features (mixed numeric/categorical). "
            "Metric: RMSE (lower is better)."
        ),
    },
    "fusion": {
        "plan": (
            "Fuse the best ideas from Branch A (CatBoost with target encoding, AUC=0.89) "
            "and Branch B (LightGBM with feature interactions, AUC=0.87). "
            "Use CatBoost as the base model but incorporate the polynomial feature interactions "
            "from Branch B. Keep the same validation split."
        ),
        "agent_context": (
            "You are a Kaggle Grandmaster performing fusion of two successful solution branches. "
            "Branch A: CatBoost with native categorical handling, AUC=0.89. "
            "Branch B: LightGBM with manual polynomial feature interactions, AUC=0.87. "
            "Your goal is to combine the best elements of both approaches."
        ),
        "parent_output": "Branch A AUC: 0.89 | Branch B AUC: 0.87",
        "child_memory": "",
        "task_desc": (
            "Binary classification on tabular data. "
            "Dataset: 10000 rows, 30 features. "
            "Metric: AUC-ROC (higher is better)."
        ),
    },
    "draft": {
        "plan": (
            "Start with an XGBoost baseline. Use one-hot encoding for categoricals. "
            "Split data 80/20 stratified. Train for 500 rounds with early stopping. "
            "Generate predictions on test set."
        ),
        "agent_context": (
            "You are a Kaggle Grandmaster drafting an initial solution for a new competition. "
            "This is the first attempt — no prior solutions exist. "
            "You need to propose a solid baseline approach."
        ),
        "parent_output": "",
        "child_memory": "",
        "task_desc": (
            "Predict loan default (binary classification). "
            "Dataset: 50000 rows, 40 features (20 categorical, 20 numeric). "
            "Metric: F1-score (higher is better). "
            "Significant class imbalance: 95% no-default, 5% default."
        ),
    },
}


def ensure_run_dir(run_name):
    """Create the run directory with a minimal journal.json so the dashboard lists it."""
    logs_dir = Path("runs") / run_name / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    journal_path = logs_dir / "journal.json"
    if not journal_path.exists():
        journal = {
            "nodes": [{
                "step": 0, "id": "socrates_test_root", "stage": "root",
                "metric": {"value": 0, "maximize": True},
                "is_buggy": None, "exec_time": 0,
                "created_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "finish_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "plan": "Socrates test run",
            }],
            "node2parent": {},
        }
        with open(journal_path, "w") as f:
            json.dump(journal, f, indent=2)

    return str(logs_dir)


def make_agent(scenario, max_rounds, log_dir):
    """Build a real-config agent for API calls."""
    code_cfg = SimpleNamespace(
        model="claude-sonnet-4-20250514", temp=0.7, base_url="", api_key="",
    )
    feedback_cfg = SimpleNamespace(
        model="claude-sonnet-4-20250514", temp=0.7, base_url="", api_key="",
    )
    acfg = SimpleNamespace(
        use_socrates_review=True,
        socrates_max_rounds=max_rounds,
        use_global_memory=False,
        feedback=feedback_cfg,
        code=code_cfg,
    )
    cfg = SimpleNamespace(
        log_dir=log_dir,
        agent=SimpleNamespace(code=code_cfg, feedback=feedback_cfg),
    )

    agent = MagicMock()
    agent.acfg = acfg
    agent.cfg = cfg
    agent.task_desc = scenario["task_desc"]
    agent.data_preview = "See task description for data details."
    agent.global_memory = None
    agent.socrates_state = None
    return agent


class LiveLogger:
    """Logs to both console and file."""

    def __init__(self, log_path):
        self.log_path = log_path
        with open(log_path, "w") as f:
            f.write("")

    def log(self, msg=""):
        print(msg)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def section(self, title):
        self.log(f"\n{'='*80}")
        self.log(f"  {title}")
        self.log(f"{'='*80}")

    def msg_block(self, label, text, max_lines=None):
        self.log(f"\n  {label}:")
        self.log(f"  {'-'*76}")
        lines = text.split('\n')
        show = lines[:max_lines] if max_lines else lines
        for line in show:
            self.log(f"  | {line}")
        if max_lines and len(lines) > max_lines:
            self.log(f"  | ... ({len(lines) - max_lines} more lines)")
        self.log(f"  {'-'*76}")


def run_test(stage, max_rounds, run_name, log_path):
    scenario = SCENARIOS[stage]

    # Set up run directory so _save_transcript writes the real JSONL
    log_dir = ensure_run_dir(run_name)
    agent = make_agent(scenario, max_rounds, log_dir)
    ll = LiveLogger(log_path)

    import agents.socrates.approval_loop as loop

    # Wrap LLM calls with logging (call the real functions, just log around them)
    _orig_agentic = loop.agentic_chat
    _orig_chat = loop.llm_chat
    socrates_call = {"n": 0}
    planner_call = {"n": 0}

    def logged_agentic_chat(**kwargs):
        socrates_call["n"] += 1
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools")

        ll.section(f"SOCRATES (round {socrates_call['n']})")
        ll.log(f"  Model: {kwargs.get('model')}")
        ll.log(f"  Tools: {[t['name'] for t in tools] if tools else 'None'}")
        ll.msg_block("Socrates receives", messages[-1]["content"])

        response, final_msgs = _orig_agentic(**kwargs)

        approved = "[APPROVED]" in response.upper()
        ll.msg_block("Socrates responds", response)
        ll.log(f"  >> Approved: {approved}")
        return response, final_msgs

    def logged_chat(**kwargs):
        planner_call["n"] += 1
        messages = kwargs.get("messages", [])

        ll.section(f"PLANNER — {stage} agent (round {planner_call['n']})")
        ll.log(f"  Model: {kwargs.get('model')}")
        ll.log(f"  System (agent context, first 200 chars):")
        ll.log(f"    {str(kwargs.get('system_message', ''))[:200]}...")
        ll.msg_block("Planner receives", messages[-1]["content"], max_lines=40)

        response = _orig_chat(**kwargs)

        ll.msg_block("Planner responds", str(response))
        return response

    loop.agentic_chat = logged_agentic_chat
    loop.llm_chat = logged_chat

    # Header
    ll.log("#" * 80)
    ll.log(f"#  SOCRATES LIVE TEST — {stage.upper()} AGENT")
    ll.log(f"#  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    ll.log("#" * 80)
    ll.log(f"\n  Stage:      {stage}")
    ll.log(f"  Max rounds: {max_rounds}")
    ll.log(f"  Task:       {scenario['task_desc'][:100]}...")
    ll.log(f"  Dashboard:  runs/{run_name}/logs/socrates_transcripts.jsonl")
    ll.msg_block("ORIGINAL PLAN", scenario["plan"])
    if scenario["parent_output"]:
        ll.log(f"\n  Parent output: {scenario['parent_output']}")
    if scenario["child_memory"]:
        ll.msg_block("Memory (previous attempts)", scenario["child_memory"])

    # Run the real Socrates review — _save_transcript writes the JSONL
    from agents.socrates import socratic_review

    start = time.time()
    final_plan = socratic_review(
        agent, scenario["plan"],
        parent_output=scenario["parent_output"],
        child_memory=scenario["child_memory"],
        agent_prompt_context=scenario["agent_context"],
        stage_name=stage,
    )
    elapsed = time.time() - start

    # Restore
    loop.agentic_chat = _orig_agentic
    loop.llm_chat = _orig_chat

    # Result
    ll.section("FINAL RESULT")
    ll.log(f"  Plan changed: {final_plan != scenario['plan']}")
    ll.log(f"  Rounds used:  {socrates_call['n']}")
    ll.log(f"  Time:         {elapsed:.1f}s")
    ll.msg_block("Final plan", final_plan)
    ll.log(f"\n  Dashboard: python dashboard.py → select '{run_name}' → Socrates Reviews tab")
    ll.log(f"  Full log:  {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Live Socrates review tester")
    parser.add_argument(
        "--stage", choices=["improve", "evolution", "fusion", "draft"],
        default="improve", help="Which agent stage to test (default: improve)"
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Max discussion rounds (default: 3)"
    )
    parser.add_argument(
        "--run", type=str, default=None,
        help="Run name for dashboard output (default: auto-generated)"
    )
    parser.add_argument(
        "--list-runs", action="store_true",
        help="List available runs and exit"
    )
    args = parser.parse_args()

    if args.list_runs:
        runs_dir = Path("runs")
        if runs_dir.exists():
            for d in sorted(runs_dir.iterdir(), reverse=True):
                journal = d / "logs" / "journal.json"
                if journal.exists():
                    print(f"  {d.name}")
        else:
            print("No runs/ directory found.")
        return

    run_name = args.run or f"socrates_test_{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join(os.path.dirname(__file__), f"socrates_live_{args.stage}.log")

    print(f"Running Socrates review for {args.stage} agent ({args.rounds} max rounds)...")
    print(f"Dashboard output → runs/{run_name}/logs/socrates_transcripts.jsonl")
    print()

    run_test(args.stage, args.rounds, run_name, log_path)


if __name__ == "__main__":
    main()
