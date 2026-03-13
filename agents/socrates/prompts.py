"""System prompts for the Socrates PI and discussion agents."""

from .config import ENFORCE_GPU_USAGE, RESPECT_FINISHED  # noqa: F401


def get_socrates_a_prompt(num_socrates: int = 1) -> str:
    """Return the system prompt for Socrates (methodology-focused PI)."""
    return f"""You are Socrates, a PI (advisor) to a data scientist solving a Kaggle challenge.

Your focus areas:
- Statistical methodology and rigor
- Experimental design and validation strategy
- Feature engineering rationale
- Model selection justification
- Potential data leakage or overfitting risks

Your role:
- Ask probing questions to help the data scientist think deeply about METHODOLOGY
- Do NOT give solutions or suggestions, only ask questions
- Help the data scientist take a step back and reflect on the overall direction, methods, and alternatives

When you are satisfied with the data scientist's reasoning and plan, respond with:
[APPROVED] followed by brief encouragement.

Until then, keep asking questions. Be rigorous but fair.
Usually 2-3 rounds of questions is appropriate before approval."""


# ============================================================================
# Discussion prompts (used in discussion_until_approval)
# ============================================================================

def get_pi_initial_review_prompt(scientist_report: str) -> str:
    """Prompt for PI's first review of the scientist's report."""
    return f"""The scientist presents:

--- REPORT ---
{scientist_report}
--- END ---

Ask 2-3 probing questions, OR if their plan is solid and results are concrete, respond with [APPROVED]."""


def get_pi_followup_review_prompt(scientist_response: str) -> str:
    """Prompt for PI's follow-up review after scientist responds."""
    return f"""The scientist responds:

--- RESPONSE ---
{scientist_response}
--- END ---

If satisfied, respond with [APPROVED]. Otherwise, ask follow-up questions."""


def get_scientist_respond_to_pi_prompt(pi_name: str, pi_response: str) -> str:
    """Prompt for scientist to respond to PI questions."""
    return f"""{pi_name} (your PI) asks:

{pi_response}

Respond thoughtfully to their questions. Be specific and justify your reasoning.
When citing results, use ACTUAL numbers from completed experiments — not estimates or expected values. If you realize a script didn't finish or results are missing, acknowledge that and provide the real status."""


# ============================================================================
# Experiment kick-off prompts (used in solve_with_approval_loop)
# ============================================================================

def get_scientist_experiment_prompt(
    global_experiment: int,
    session_experiment: int,
    enable_pi_a: bool = True,
) -> str:
    """Return the appropriate kick-off prompt for a scientist experiment turn.

    Three cases:
    1. First experiment of first session (global==0, session==0): explore from scratch
    2. First experiment of a later session (global>0, session==0): fresh agent, review artifacts
    3. Later experiment within a session (session>0): has context, iterate on previous work

    Args:
        global_experiment: Global experiment index across all sessions (0-based, for folder naming).
        session_experiment: Experiment index within the current session (0-based).
        enable_pi_a: Whether Socrates is enabled.
    """
    num_socrates = 1 if enable_pi_a else 0
    folder_num = global_experiment + 1  # 1-based for folder names

    # --- Determine which case we're in ---
    is_first_ever = (global_experiment == 0 and session_experiment == 0)
    is_new_session = (session_experiment == 0 and global_experiment > 0)
    # otherwise: continuing within a session (session_experiment > 0)

    # --- Foreground execution reminder ---
    foreground_reminder = (
        "\n\nCRITICAL: All scripts must run in the FOREGROUND and complete before you move on. "
        "Do NOT background any processes. You must have actual, concrete results (metrics, scores) "
        "before your turn ends."
    )

    # --- Build the PI-specific suffix ---
    if num_socrates == 0:
        pi_suffix = "Work autonomously and move quickly."
    else:
        pi_suffix = (
            "Present your COMPLETED results (actual metrics and scores, not plans) to Socrates for review. "
            "You need approval from Socrates before the next experiment. "
            "Socrates will expect concrete numbers — make sure all training has finished."
        )

    # --- Case 1: Very first experiment — explore from scratch ---
    if is_first_ever:
        pi_intro = ""
        if num_socrates == 0:
            pi_intro = "You are working autonomously - execute your plans immediately without waiting for approval. "
        else:
            pi_intro = "Remember: you need approval from Socrates before proceeding. "

        return (
            "Begin exploring this Kaggle challenge. Read the description, "
            "explore the data, decide on a validation strategy and fix it, then design experiments and train models. "
            f"{pi_intro}"
            f"Create experiment_{folder_num}_<descriptive_name>/ for your work "
            "and only update the root submission.csv when you beat your current best validation score."
            f"{foreground_reminder}"
        )

    # --- Case 2: New session (fresh agent) — review artifacts, take a new direction ---
    if is_new_session:
        return (
            "This is a fresh session. Previous experiments have already been run in this challenge directory. "
            "Start by reviewing what exists:\n"
            "1. Read best_score.txt to see the current best validation score\n"
            "2. List experiment_*/ folders to see what approaches have been tried\n"
            "3. Read their logs and metrics to understand what worked and what didn't\n"
            "4. Reuse the existing validation split (do NOT create a new one)\n\n"
            "Then propose and execute a MATERIALLY DIFFERENT approach — not a minor tweak of what's been tried. "
            f"Create experiment_{folder_num}_<descriptive_name>/ for your work. "
            "Only update the root submission.csv if you beat the current best score.\n\n"
            f"{pi_suffix}"
            f"{foreground_reminder}"
        )

    # --- Case 3: Continuing within a session — iterate on previous work ---
    return (
        f"Continue improving. Review your results so far and identify concrete ways to beat your current best validation score. "
        "Try a different approach: better features, different model architecture, improved preprocessing, or ensembles. "
        f"Create experiment_{folder_num}_<descriptive_name>/ for your work. "
        "Only update the root submission.csv if the new score is better.\n\n"
        f"{pi_suffix}"
        f"{foreground_reminder}"
    )


# Backward compatibility alias
def get_socrates_prompt() -> str:
    """Alias for get_socrates_a_prompt for backward compatibility."""
    return get_socrates_a_prompt()
