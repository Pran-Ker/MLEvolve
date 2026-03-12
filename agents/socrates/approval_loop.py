"""Discussion loop between Scientist and PI (Socrates) until approval."""

import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, TextIO
import json
from datetime import datetime
import sys

from .providers import (
    AgentOptions,
    AgentClient,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ClaudeAgentClient,
    PydanticAgentClient,
    OpenHandsAgentClient,
)
from .config import (
    CHALLENGE_PATH,
    SOCRATES_PYTHON_BIN,
    MAX_SESSIONS,
    EXPERIMENTS_PER_SESSION,
    MAX_DISCUSSION_ROUNDS,
    MAX_TURNS,
    PI_MODE,
    RESPECT_FINISHED,
    ENABLE_PI_A,
    SCIENTIST_PROVIDER,
    SCIENTIST_MODEL,
    SOCRATES_A_PROVIDER,
    SOCRATES_A_MODEL,
    ProviderType,
)
from .prompts import (
    get_scientist_prompt,
    get_socrates_a_prompt,
    get_pi_initial_review_prompt,
    get_pi_followup_review_prompt,
    get_scientist_respond_to_pi_prompt,
    get_scientist_experiment_prompt,
)


class TeeLogger:
    """Logger that writes to both stdout and a file."""

    def __init__(self, log_file: TextIO):
        self.log_file = log_file
        self.stdout = sys.stdout

    def print(self, *args, **kwargs):
        """Print to both stdout and log file."""
        print(*args, **kwargs, file=self.stdout)
        print(*args, **kwargs, file=self.log_file)
        self.log_file.flush()


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


@asynccontextmanager
async def _dummy_context() -> AsyncIterator[None]:
    """Dummy context manager for when PIs are disabled."""
    yield None


async def _collect_response(agent: AgentClient, logger: TeeLogger, prefix: str = "💬") -> str:
    """Collect text from an agent's response stream, logging as it arrives.

    Console output is truncated to 300 chars per block; log file gets full text.
    """
    text = ""
    async for msg in agent.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text += block.text + "\n"
                    # Truncated for console
                    display = block.text[:300] + "..." if len(block.text) > 300 else block.text
                    print(f"   {prefix} {display}", file=logger.stdout)
                    # Full text in log file
                    print(f"   {prefix} {block.text}", file=logger.log_file)
                    logger.log_file.flush()
                elif isinstance(block, ToolUseBlock):
                    logger.print(f"   🔧 {block.name}")
    return text


def create_client(provider: ProviderType, options: AgentOptions) -> AgentClient:
    """Create an agent client based on the specified provider."""
    if provider == "pydantic":
        return PydanticAgentClient(options=options)
    elif provider == "openhands":
        return OpenHandsAgentClient(options=options)
    else:
        return ClaudeAgentClient(options=options)


def _read_best_score(challenge_path: Path) -> tuple[float | None, str | None]:
    """
    Read the best score from best_score.txt.

    Returns:
        (score, experiment_name) or (None, None) if file doesn't exist or can't be parsed
    """
    best_score_file = challenge_path / "best_score.txt"
    if not best_score_file.exists():
        return None, None

    try:
        content = best_score_file.read_text().strip()
        # Expected format: "0.85 from experiment_3_xgboost"
        # or just "0.85"
        parts = content.split()
        if len(parts) >= 1:
            try:
                score = float(parts[0])
                experiment = " ".join(parts[2:]) if len(parts) > 2 else None
                return score, experiment
            except ValueError:
                return None, None
    except Exception:
        pass

    return None, None


def _ensure_submission(challenge_path: Path, logger: TeeLogger) -> None:
    """
    Safety net: verify that submission.csv exists at the challenge root.

    If the agent placed it inside an experiment folder but not at the root,
    copy the one from the highest-numbered experiment folder as a fallback.
    If no submission exists anywhere, log an error (honest zero).
    """
    root_submission = challenge_path / "submission.csv"

    if root_submission.exists() and root_submission.stat().st_size > 0:
        logger.print(f"\n✅ submission.csv found at challenge root ({root_submission.stat().st_size} bytes)")
        return

    # Root submission missing — scan experiment folders for a fallback
    logger.print("\n⚠️  No submission.csv at challenge root — scanning experiment folders...")

    found = []
    for d in challenge_path.iterdir():
        if d.is_dir() and d.name.startswith("experiment_"):
            candidate = d / "submission.csv"
            if candidate.exists() and candidate.stat().st_size > 0:
                found.append((d.name, candidate))

    if len(found) == 1:
        exp_name, candidate = found[0]
        shutil.copy2(candidate, root_submission)
        logger.print(f"   📋 Recovered submission.csv from {exp_name} → challenge root")
    elif len(found) > 1:
        names = ", ".join(name for name, _ in sorted(found))
        logger.print(f"   ❌ Multiple experiment folders contain submission.csv ({names}). Cannot determine best — skipping recovery.")
    else:
        logger.print("   ❌ No submission.csv found in any experiment folder. This run will score zero.")


async def discussion_until_approval(
    scientist: AgentClient,
    pi: AgentClient,
    scientist_report: str,
    logger: TeeLogger,
    pi_name: str = "Socrates",
    pi_emoji: str = "🏛️",
    max_rounds: int = 5,
) -> tuple[bool, int]:
    """
    Inner loop: Scientist and a PI discuss until PI approves.

    Returns:
        (approved: bool, rounds_used: int)
    """
    scientist_response = ""

    for round_num in range(max_rounds):
        logger.print(f"\n   📣 Discussion with {pi_name} - round {round_num + 1}")

        # --- PI asks questions (or approves) ---
        if round_num == 0:
            await pi.query(get_pi_initial_review_prompt(scientist_report))
        else:
            await pi.query(get_pi_followup_review_prompt(scientist_response))

        pi_response = await _collect_response(pi, logger, prefix=f"{pi_emoji} {pi_name}:")

        # Check for approval
        if "[APPROVED]" in pi_response.upper():
            logger.print(f"\n   ✅ {pi_name} APPROVED after {round_num + 1} round(s)")
            return True, round_num + 1

        # --- Scientist responds to questions ---
        await scientist.query(get_scientist_respond_to_pi_prompt(pi_name, pi_response))

        scientist_response = await _collect_response(scientist, logger, prefix="🔬 Scientist:")

    logger.print(f"\n   ⚠️ Max rounds with {pi_name} reached without approval")
    return False, max_rounds


async def solve_with_approval_loop() -> None:
    """
    Main loop with explicit approval mechanism.

    Two modes:
    1. PI mode: Fresh scientist per experiment, persistent Socrates PI
    2. Non-PI mode: One scientist session until [FINISHED], no Socrates agent
    """
    # Create log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = CHALLENGE_PATH / f"socrates_run_{timestamp}.log"

    with open(log_file_path, 'w') as log_file:
        logger = TeeLogger(log_file)
        logger.print(f"📝 Logging to: {log_file_path}")
        logger.print("")

        await _run_with_logging(logger)


async def _run_with_logging(logger: TeeLogger) -> None:
    """Main loop implementation with logging."""
    run_start = time.monotonic()
    num_socrates = 1 if ENABLE_PI_A else 0

    # --- Log full configuration header ---
    logger.print(f"{'='*60}")
    logger.print("SOCRATES RUN CONFIGURATION")
    logger.print(f"{'='*60}")
    logger.print(f"  Timestamp:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.print(f"  Challenge path:         {CHALLENGE_PATH}")
    logger.print(f"  PI mode:                {PI_MODE}")
    logger.print(f"  Scientist:              provider={SCIENTIST_PROVIDER}, model={SCIENTIST_MODEL}")
    if ENABLE_PI_A:
        logger.print(f"  Socrates:               provider={SOCRATES_A_PROVIDER}, model={SOCRATES_A_MODEL}")
    else:
        logger.print(f"  Socrates:               DISABLED")
    logger.print(f"  Max turns per exp:      {MAX_TURNS}")
    if PI_MODE:
        logger.print(f"  Max sessions:           {MAX_SESSIONS}")
        logger.print(f"  Experiments/session:    {EXPERIMENTS_PER_SESSION}")
        logger.print(f"  Total experiments:      {MAX_SESSIONS * EXPERIMENTS_PER_SESSION}")
        logger.print(f"  Max discussion rounds:  {MAX_DISCUSSION_ROUNDS}")
        logger.print(f"  Respect [FINISHED]:     {RESPECT_FINISHED}")
    logger.print(f"  Python bin:             {SOCRATES_PYTHON_BIN}")
    logger.print(f"{'='*60}\n")

    options_scientist = AgentOptions(
        system_prompt=get_scientist_prompt(CHALLENGE_PATH, num_socrates, pi_mode=PI_MODE),
        allowed_tools=["Read", "Write", "Bash", "Glob"],
        permission_mode="acceptEdits",
        cwd=str(CHALLENGE_PATH),
        max_turns=MAX_TURNS,
        env={"PATH": f"{SOCRATES_PYTHON_BIN}:{os.environ.get('PATH', '')}"},
        model=SCIENTIST_MODEL,
    )

    options_socrates_a = AgentOptions(
        system_prompt=get_socrates_a_prompt(num_socrates),
        allowed_tools=[],
        max_turns=5,
        model=SOCRATES_A_MODEL,
    )

    if not PI_MODE:
        # NON-PI MODE: Single scientist session, no Socrates agents
        logger.print(f"🔄 NON-PI MODE: Running until [FINISHED] signal")
        async with create_client(SCIENTIST_PROVIDER, options_scientist) as scientist:
            await _run_non_pi_mode(scientist, logger)
    else:
        # PI MODE: Outer loop (fresh scientist per session), inner loop (experiments within session)
        total = MAX_SESSIONS * EXPERIMENTS_PER_SESSION
        logger.print(f"🔢 PI MODE: {MAX_SESSIONS} sessions x {EXPERIMENTS_PER_SESSION} experiments = {total} total")
        socrates_a_ctx = create_client(SOCRATES_A_PROVIDER, options_socrates_a) if ENABLE_PI_A else None

        async with socrates_a_ctx if socrates_a_ctx else _dummy_context() as socrates_a:
            await _run_pi_mode(options_scientist, socrates_a, logger)

    # Safety net: ensure a submission.csv exists at the challenge root
    _ensure_submission(CHALLENGE_PATH, logger)

    run_elapsed = time.monotonic() - run_start
    logger.print(f"\n✅ Experiment loop complete! (total time: {_format_duration(run_elapsed)})")


async def _run_non_pi_mode(scientist: AgentClient, logger: TeeLogger):
    """Non-PI mode: Single scientist session until [FINISHED]."""
    session_start = time.monotonic()

    logger.print("\n" + "=" * 60)
    logger.print("NON-PI SESSION")
    logger.print("=" * 60)

    logger.print("\n🔬 SCIENTIST working...")

    await scientist.query(get_scientist_experiment_prompt(
        global_experiment=0, session_experiment=0,
        enable_pi_a=ENABLE_PI_A,
    ))

    scientist_report = await _collect_response(scientist, logger, prefix="💬")

    session_elapsed = time.monotonic() - session_start

    if "[FINISHED]" in scientist_report.upper():
        logger.print(f"\n🏁 Scientist signaled [FINISHED] — achieved best score.")
    else:
        logger.print(f"\n⚠️ Scientist stopped without [FINISHED] signal")

    logger.print(f"⏱️  Session duration: {_format_duration(session_elapsed)}")

    # Log final score
    final_score, experiment_name = _read_best_score(CHALLENGE_PATH)
    if final_score is not None:
        logger.print(f"\n📊 Final best score: {final_score}")
        if experiment_name:
            logger.print(f"   From: {experiment_name}")

        # Save summary for non-PI mode
        summary_file = CHALLENGE_PATH / "score_history.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "mode": "non-pi",
                "final_best_score": final_score,
                "experiment_name": experiment_name,
                "finished_signal": "[FINISHED]" in scientist_report.upper(),
                "duration_seconds": round(session_elapsed, 1)
            }, f, indent=2)
        logger.print(f"📝 Score summary saved to: {summary_file}")


async def _run_pi_mode(
    scientist_options: AgentOptions,
    socrates_a,
    logger: TeeLogger,
):
    """
    PI mode with two loops:

    Outer loop (sessions): Creates a fresh scientist each time. The new scientist
    reads experiment folders and best_score.txt from previous sessions to understand
    what has been tried, then takes a new approach.

    Inner loop (experiments): Within a session the scientist keeps its conversation
    context, iterating on its own work across multiple experiments.

    Socrates agent persists across all sessions to accumulate trajectory knowledge.
    """
    global_experiment = 0  # globally sequential experiment counter (for folder naming)

    # Track score progression for analysis
    # We detect improvements by checking when best_score.txt changes, rather than
    # comparing scores with > or <, since some metrics are higher-is-better (accuracy)
    # and others are lower-is-better (RMSE, log loss).
    score_history = []
    last_known_score = None  # last score read from best_score.txt

    for session in range(MAX_SESSIONS):
        session_start = time.monotonic()

        # Read best score at start of session
        score_at_session_start, _ = _read_best_score(CHALLENGE_PATH)

        logger.print(f"\n{'='*60}")
        logger.print(f"SESSION {session + 1} of {MAX_SESSIONS} (fresh scientist)")
        if score_at_session_start is not None:
            logger.print(f"📊 Current best score: {score_at_session_start}")
        logger.print("=" * 60)

        async with create_client(SCIENTIST_PROVIDER, scientist_options) as scientist:
            for experiment in range(EXPERIMENTS_PER_SESSION):
                exp_start = time.monotonic()

                logger.print(f"\n  {'─'*56}")
                logger.print(f"  Experiment {experiment + 1}/{EXPERIMENTS_PER_SESSION} "
                      f"(global #{global_experiment + 1}, session {session + 1})")
                logger.print(f"  {'─'*56}")

                # --- Scientist does work ---
                logger.print("\n🔬 SCIENTIST working...")

                await scientist.query(
                    get_scientist_experiment_prompt(
                        global_experiment=global_experiment,
                        session_experiment=experiment,
                        enable_pi_a=ENABLE_PI_A,
                    )
                )

                scientist_report = await _collect_response(scientist, logger, prefix="💬")

                scientist_elapsed = time.monotonic() - exp_start
                logger.print(f"\n⏱️  Scientist work: {_format_duration(scientist_elapsed)}")

                # --- Check for [FINISHED] signal ---
                if RESPECT_FINISHED and "[FINISHED]" in scientist_report.upper():
                    logger.print(f"\n🏁 Scientist signaled [FINISHED] — ending session {session + 1} early")
                    global_experiment += 1
                    # Track score before breaking
                    current_score, experiment_name = _read_best_score(CHALLENGE_PATH)
                    if current_score is not None:
                        improved = last_known_score is None or current_score != last_known_score
                        if improved:
                            last_known_score = current_score
                            logger.print(f"\n🎯 NEW BEST SCORE: {current_score} (from {experiment_name or 'unknown'})")
                        score_history.append({
                            "global_experiment": global_experiment,
                            "session": session + 1,
                            "experiment": experiment + 1,
                            "score": current_score,
                            "experiment_name": experiment_name,
                            "improved": improved,
                            "approved_a": None,
                            "rounds_a": None,
                            "duration_seconds": round(scientist_elapsed, 1),
                            "finished_signal": True
                        })
                    break

                # --- Discussion with Socrates (methodology focus) ---
                approved_a = True
                rounds_a = 0
                if ENABLE_PI_A:
                    discussion_start = time.monotonic()
                    logger.print("\n📣 DISCUSSION PHASE - Socrates (Methodology)")
                    approved_a, rounds_a = await discussion_until_approval(
                        scientist, socrates_a, scientist_report, logger,
                        pi_name="Socrates", pi_emoji="🏛️",
                        max_rounds=MAX_DISCUSSION_ROUNDS
                    )

                    if not approved_a:
                        logger.print("⚠️ Could not get Socrates approval, continuing anyway...")

                    discussion_elapsed = time.monotonic() - discussion_start
                    logger.print(f"⏱️  Socrates discussion: {_format_duration(discussion_elapsed)} ({rounds_a} round(s))")

                # Summary
                if not ENABLE_PI_A:
                    logger.print("\n✅ Baseline-PI mode - proceeding to next experiment")
                elif approved_a:
                    logger.print(f"\n✅ Socrates APPROVED - proceeding to next experiment")
                else:
                    logger.print(f"\n⚠️ Socrates did not approve, but continuing...")

                global_experiment += 1
                exp_elapsed = time.monotonic() - exp_start
                logger.print(f"⏱️  Total experiment time: {_format_duration(exp_elapsed)}")

                # --- Track score progression after experiment ---
                # Detect improvement by checking if the score in best_score.txt changed.
                # The scientist only overwrites best_score.txt when it finds a better
                # result, so any change means an improvement (regardless of metric direction).
                current_score, experiment_name = _read_best_score(CHALLENGE_PATH)
                improved = False
                if current_score is not None:
                    if last_known_score is None or current_score != last_known_score:
                        improved = True
                        last_known_score = current_score
                        logger.print(f"\n🎯 NEW BEST SCORE: {current_score} (from {experiment_name or 'unknown'})")

                    score_history.append({
                        "global_experiment": global_experiment,
                        "session": session + 1,
                        "experiment": experiment + 1,
                        "score": current_score,
                        "experiment_name": experiment_name,
                        "improved": improved,
                        "approved_a": approved_a if ENABLE_PI_A else None,
                        "rounds_a": rounds_a if ENABLE_PI_A else None,
                        "duration_seconds": round(exp_elapsed, 1),
                        "finished_signal": False
                    })

        # Scientist context closed — next session gets a fresh one
        session_elapsed = time.monotonic() - session_start
        session_end_score, _ = _read_best_score(CHALLENGE_PATH)

        if score_at_session_start is not None and session_end_score is not None:
            if session_end_score != score_at_session_start:
                logger.print(f"\n✨ Session {session + 1} CHANGED score: {score_at_session_start} → {session_end_score}")
            else:
                logger.print(f"\n📊 Session {session + 1} final score: {session_end_score} (unchanged)")
        elif session_end_score is not None:
            logger.print(f"\n📊 Session {session + 1} final score: {session_end_score}")

        logger.print(f"⏱️  Session {session + 1} duration: {_format_duration(session_elapsed)}")
        logger.print(f"🔄 Session {session + 1} complete — scientist context reset")

    logger.print(f"\n🏁 PI mode complete - ran {global_experiment} experiments across {MAX_SESSIONS} sessions")

    # Print score progression analysis
    if score_history:
        logger.print(f"\n{'='*60}")
        logger.print("📈 SCORE PROGRESSION ANALYSIS")
        logger.print("=" * 60)

        improvements_count = sum(1 for entry in score_history if entry["improved"])
        logger.print(f"\nTotal improvements: {improvements_count}/{len(score_history)} experiments")

        # Group by session
        improvements_by_session = {}
        for entry in score_history:
            sess = entry["session"]
            if sess not in improvements_by_session:
                improvements_by_session[sess] = 0
            if entry["improved"]:
                improvements_by_session[sess] += 1

        logger.print("\nImprovements per session:")
        for sess in sorted(improvements_by_session.keys()):
            count = improvements_by_session[sess]
            logger.print(f"  Session {sess}: {count} improvement(s)")

        # Count improvements from later sessions (session > 1)
        later_session_improvements = sum(
            1 for entry in score_history
            if entry["improved"] and entry["session"] > 1
        )
        logger.print(f"\nImprovements from later sessions (2+): {later_session_improvements}")

        # Discussion stats
        if ENABLE_PI_A:
            total_rounds_a = sum(e["rounds_a"] or 0 for e in score_history)
            approvals_a = sum(1 for e in score_history if e.get("approved_a") is True)
            logger.print(f"\nDiscussion stats:")
            logger.print(f"  Socrates: {approvals_a}/{len(score_history)} approved, {total_rounds_a} total rounds")

        # Final best score
        final_score, final_exp = _read_best_score(CHALLENGE_PATH)
        if final_score is not None:
            logger.print(f"\nFinal best score: {final_score}")
            best_entry = next((e for e in reversed(score_history) if e["improved"]), None)
            if best_entry:
                logger.print(f"  Achieved in: Session {best_entry['session']}, "
                      f"Experiment {best_entry['experiment']} "
                      f"(global #{best_entry['global_experiment']})")

        # Save detailed history to JSON
        history_file = CHALLENGE_PATH / "score_history.json"
        with open(history_file, 'w') as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "config": {
                    "pi_mode": PI_MODE,
                    "scientist_provider": SCIENTIST_PROVIDER,
                    "scientist_model": SCIENTIST_MODEL,
                    "socrates_a_provider": SOCRATES_A_PROVIDER if ENABLE_PI_A else None,
                    "socrates_a_model": SOCRATES_A_MODEL if ENABLE_PI_A else None,
                    "max_sessions": MAX_SESSIONS,
                    "experiments_per_session": EXPERIMENTS_PER_SESSION,
                    "max_discussion_rounds": MAX_DISCUSSION_ROUNDS,
                    "max_turns": MAX_TURNS,
                },
                "total_experiments": len(score_history),
                "total_improvements": improvements_count,
                "later_session_improvements": later_session_improvements,
                "final_best_score": final_score,
                "history": score_history
            }, f, indent=2)
        logger.print(f"\n📝 Detailed score history saved to: {history_file}")
