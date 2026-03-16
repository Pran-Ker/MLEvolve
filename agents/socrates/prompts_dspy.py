"""DSPy-optimized prompts for the Socrates PI and discussion agents."""

import dspy

from .config import ENFORCE_GPU_USAGE, RESPECT_FINISHED  # noqa: F401


# ============================================================================
# Signatures
# ============================================================================

class SocratesReview(dspy.Signature):
    """You are Socrates, a PI (advisor) to a data scientist solving a Kaggle challenge.

    Focus areas: statistical methodology and rigor, experimental design and validation strategy,
    feature engineering rationale, model selection justification, potential data leakage or overfitting risks.

    Ask probing questions to help the data scientist think deeply about METHODOLOGY.
    Do NOT give solutions or suggestions, only ask questions.
    Help the data scientist take a step back and reflect on the overall direction, methods, and alternatives.

    When satisfied with the data scientist's reasoning and plan, respond with [APPROVED] followed by brief encouragement.
    Until then, keep asking questions. Be rigorous but fair. Usually 2-3 rounds of questions is appropriate before approval."""

    scientist_report: str = dspy.InputField(desc="The scientist's report or response to review")
    is_followup: bool = dspy.InputField(desc="Whether this is a follow-up review (True) or initial review (False)")
    review: str = dspy.OutputField(desc="2-3 probing questions, OR [APPROVED] if the plan is solid and results are concrete")


class ScientistRespondToPI(dspy.Signature):
    """Respond thoughtfully to your PI's questions. Be specific and justify your reasoning.
    When citing results, use ACTUAL numbers from completed experiments — not estimates or expected values.
    If you realize a script didn't finish or results are missing, acknowledge that and provide the real status."""

    pi_name: str = dspy.InputField(desc="Name of the PI asking questions")
    pi_questions: str = dspy.InputField(desc="The PI's questions to respond to")
    response: str = dspy.OutputField(desc="Specific, well-justified responses to each question with actual metrics")


class ExploreChallenge(dspy.Signature):
    """Begin exploring a Kaggle challenge from scratch. Read the description, explore the data,
    decide on a validation strategy and fix it, then design experiments and train models.

    CRITICAL: All scripts must run in the FOREGROUND and complete before you move on.
    Do NOT background any processes. You must have actual, concrete results (metrics, scores)
    before your turn ends."""

    folder_name: str = dspy.InputField(desc="Experiment folder name to create, e.g. experiment_1_<descriptive_name>/")
    has_pi: bool = dspy.InputField(desc="Whether a PI advisor is enabled for approval")
    plan: str = dspy.OutputField(desc="Exploration plan and executed results with concrete metrics")


class ReviewAndIterate(dspy.Signature):
    """This is a fresh session. Previous experiments have already been run in this challenge directory.

    Start by reviewing what exists:
    1. Read best_score.txt to see the current best validation score
    2. List experiment_*/ folders to see what approaches have been tried
    3. Read their logs and metrics to understand what worked and what didn't
    4. Reuse the existing validation split (do NOT create a new one)

    Then propose and execute a MATERIALLY DIFFERENT approach — not a minor tweak of what's been tried.
    Only update the root submission.csv if you beat the current best score.

    CRITICAL: All scripts must run in the FOREGROUND and complete before you move on.
    Do NOT background any processes. You must have actual, concrete results (metrics, scores)
    before your turn ends."""

    folder_name: str = dspy.InputField(desc="Experiment folder name to create")
    pi_instruction: str = dspy.InputField(desc="Instructions regarding PI approval workflow")
    plan_and_results: str = dspy.OutputField(desc="New approach with executed results and concrete metrics")


class ContinueImproving(dspy.Signature):
    """Continue improving. Review your results so far and identify concrete ways to beat your
    current best validation score. Try a different approach: better features, different model
    architecture, improved preprocessing, or ensembles.
    Only update the root submission.csv if the new score is better.

    CRITICAL: All scripts must run in the FOREGROUND and complete before you move on.
    Do NOT background any processes. You must have actual, concrete results (metrics, scores)
    before your turn ends."""

    folder_name: str = dspy.InputField(desc="Experiment folder name to create")
    pi_instruction: str = dspy.InputField(desc="Instructions regarding PI approval workflow")
    plan_and_results: str = dspy.OutputField(desc="Improved approach with executed results and concrete metrics")


# ============================================================================
# Modules
# ============================================================================

class SocratesPI(dspy.Module):
    """Socrates PI module — reviews scientist work and asks probing questions."""

    def __init__(self):
        self.review = dspy.ChainOfThought(SocratesReview)

    def forward(self, scientist_report: str, is_followup: bool = False):
        return self.review(scientist_report=scientist_report, is_followup=is_followup)


class ScientistResponder(dspy.Module):
    """Scientist module — responds to PI questions with evidence."""

    def __init__(self):
        self.respond = dspy.ChainOfThought(ScientistRespondToPI)

    def forward(self, pi_name: str, pi_questions: str):
        return self.respond(pi_name=pi_name, pi_questions=pi_questions)


class ScientistExperiment(dspy.Module):
    """Scientist module — runs an experiment based on the current state."""

    def __init__(self):
        self.explore = dspy.ChainOfThought(ExploreChallenge)
        self.review_iterate = dspy.ChainOfThought(ReviewAndIterate)
        self.continue_improving = dspy.ChainOfThought(ContinueImproving)

    def forward(
        self,
        global_experiment: int,
        session_experiment: int,
        enable_pi_a: bool = True,
    ):
        num_socrates = 1 if enable_pi_a else 0
        folder_num = global_experiment + 1
        folder_name = f"experiment_{folder_num}_<descriptive_name>/"

        if num_socrates == 0:
            pi_instruction = "Work autonomously and move quickly."
        else:
            pi_instruction = (
                "Present your COMPLETED results (actual metrics and scores, not plans) to Socrates for review. "
                "You need approval from Socrates before the next experiment. "
                "Socrates will expect concrete numbers — make sure all training has finished."
            )

        is_first_ever = (global_experiment == 0 and session_experiment == 0)
        is_new_session = (session_experiment == 0 and global_experiment > 0)

        if is_first_ever:
            return self.explore(folder_name=folder_name, has_pi=(num_socrates > 0))
        elif is_new_session:
            return self.review_iterate(folder_name=folder_name, pi_instruction=pi_instruction)
        else:
            return self.continue_improving(folder_name=folder_name, pi_instruction=pi_instruction)


# ============================================================================
# Backward-compatible string prompt API (drop-in for existing callers)
# ============================================================================

def get_socrates_a_prompt(num_socrates: int = 1) -> str:
    """Return the system prompt for Socrates (methodology-focused PI)."""
    sig = SocratesReview
    return sig.__doc__


def get_pi_initial_review_prompt(scientist_report: str) -> str:
    pi = SocratesPI()
    return pi(scientist_report=scientist_report, is_followup=False).review


def get_pi_followup_review_prompt(scientist_response: str) -> str:
    pi = SocratesPI()
    return pi(scientist_report=scientist_response, is_followup=True).review


def get_scientist_respond_to_pi_prompt(pi_name: str, pi_response: str) -> str:
    responder = ScientistResponder()
    return responder(pi_name=pi_name, pi_questions=pi_response).response


def get_scientist_experiment_prompt(
    global_experiment: int,
    session_experiment: int,
    enable_pi_a: bool = True,
) -> str:
    module = ScientistExperiment()
    result = module(
        global_experiment=global_experiment,
        session_experiment=session_experiment,
        enable_pi_a=enable_pi_a,
    )
    # The active signature's output field name varies by case
    for field in ("plan", "plan_and_results"):
        if hasattr(result, field):
            return getattr(result, field)
    return str(result)


def get_socrates_prompt() -> str:
    """Alias for get_socrates_a_prompt for backward compatibility."""
    return get_socrates_a_prompt()
