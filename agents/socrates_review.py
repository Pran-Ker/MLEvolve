"""Socrates review loop for improvement plans.

Faithful adaptation of the Socrates-EM dual-loop architecture for MLEvolve's
planning pipeline. Key architectural properties (matching the reference):

1. Socrates (PI) maintains PERSISTENT conversation history across ALL review
   calls — accumulates trajectory knowledge over the entire search.
2. Planner (scientist) gets fresh context each review call — resets per session.
3. Multi-turn discussion within each review round — Socrates sees its own prior
   questions and the planner's responses (not single-turn isolated calls).
4. [APPROVED] gate before plan proceeds to coding.

Inserted between generate_initial_plan() and refine_plan_to_json() in the
improve agent's two-stage planning pipeline.
"""

import logging

from llm import chat as llm_chat

logger = logging.getLogger("MLEvolve")


# ---------------------------------------------------------------------------
# System prompts (matching reference socrates/prompts.py)
# ---------------------------------------------------------------------------

SOCRATES_A_SYSTEM = (
    "You are Socrates A, a PI (advisor) to a machine learning planning agent "
    "solving a Kaggle challenge.\n\n"
    "Your focus areas:\n"
    "- Statistical methodology and rigor\n"
    "- Experimental design and validation strategy\n"
    "- Feature engineering rationale\n"
    "- Model selection justification\n"
    "- Potential data leakage or overfitting risks\n"
    "- Whether the proposed improvement is meaningfully different from "
    "past attempts\n\n"
    "Your role:\n"
    "- Ask probing questions to help the planning agent think deeply "
    "about METHODOLOGY\n"
    "- Do NOT give solutions or suggestions, only ask questions\n"
    "- Help the agent take a step back and reflect on the overall "
    "direction, methods, and alternatives\n\n"
    "When you are satisfied with the reasoning and plan, respond with:\n"
    "[APPROVED] followed by brief encouragement.\n\n"
    "Until then, keep asking questions. Be rigorous but fair.\n"
    "Usually 2-3 rounds of questions is appropriate before approval."
)

PLANNER_RESPOND_SYSTEM = (
    "You are a Kaggle Grandmaster planning improvements to a machine "
    "learning solution. Socrates (your methodology reviewer) is "
    "questioning your improvement plan. Respond thoughtfully to their "
    "questions. Be specific and justify your reasoning. When citing "
    "results, use ACTUAL numbers from completed experiments — not "
    "estimates or expected values. If Socrates raises valid concerns, "
    "revise your plan accordingly.\n\n"
    "End your response with your final improvement plan (possibly revised)."
)


# ---------------------------------------------------------------------------
# Prompt builders (matching reference socrates/prompts.py structure)
# ---------------------------------------------------------------------------

def _pi_initial_review_prompt(plan_text, task_desc, parent_output, child_memory):
    """Prompt for PI's first review of the planner's report.

    Matches reference get_pi_initial_review_prompt().
    """
    parts = [
        "The planning agent presents:\n",
        "--- PLAN ---",
        plan_text,
        "--- END PLAN ---\n",
        f"Task: {task_desc}\n",
    ]
    if parent_output and str(parent_output).strip():
        parts.append(f"Previous execution output:\n{parent_output}\n")
    if child_memory and str(child_memory).strip():
        parts.append(f"Previous attempts:\n{child_memory}\n")
    parts.append(
        "First, verify the plan is methodologically sound and proposes a "
        "meaningfully different approach from previous attempts.\n"
        "Then ask 2-3 probing questions about the methodology, "
        "OR if the plan is solid, respond with [APPROVED]."
    )
    return "\n".join(parts)


def _pi_followup_review_prompt(planner_response):
    """Prompt for PI's follow-up review after planner responds.

    Matches reference get_pi_followup_review_prompt().
    """
    return (
        "The planning agent responds:\n\n"
        "--- RESPONSE ---\n"
        f"{planner_response}\n"
        "--- END ---\n\n"
        "If satisfied with their reasoning and revised plan, respond "
        "with [APPROVED].\nOtherwise, ask follow-up questions."
    )


def _scientist_respond_prompt(pi_response, plan_text, task_desc):
    """Prompt for planner to respond to PI questions.

    Matches reference get_scientist_respond_to_pi_prompt().
    """
    return (
        "Socrates (your methodology reviewer) asks:\n\n"
        f"{pi_response}\n\n"
        f"Your current plan:\n{plan_text}\n\n"
        f"Task: {task_desc}\n\n"
        "Respond thoughtfully to their questions. Be specific and "
        "justify your reasoning.\n"
        "If Socrates raises valid concerns, revise your plan accordingly.\n"
        "End your response with your final (possibly revised) "
        "improvement plan."
    )


# ---------------------------------------------------------------------------
# Persistent Socrates state (mirrors reference: PI persists across sessions)
# ---------------------------------------------------------------------------

class SocratesState:
    """Persistent state for Socrates reviewer across all review calls.

    Mirrors the reference architecture where Socrates agents persist across
    all sessions while the scientist (planner) resets each session.

    In the reference:
    - Socrates A/B are created once and live across all sessions
    - Their conversation context accumulates trajectory knowledge
    - The scientist gets a fresh context each session

    Here:
    - pi_messages accumulates across all review_plan() calls
    - Each review_plan() call creates fresh planner context (scientist reset)
    """

    def __init__(self):
        self.pi_messages = []  # Socrates A conversation history (persists)
        self.total_reviews = 0
        self.total_approvals = 0
        self.total_rounds = 0

    def summary(self):
        avg_rounds = self.total_rounds / max(self.total_reviews, 1)
        return (
            f"Reviews: {self.total_reviews}, "
            f"Approvals: {self.total_approvals}/{self.total_reviews}, "
            f"Avg rounds: {avg_rounds:.1f}"
        )


# ---------------------------------------------------------------------------
# Core discussion loop (matching reference discussion_until_approval)
# ---------------------------------------------------------------------------

def discussion_until_approval(
    pi_messages,
    plan_text,
    task_desc,
    parent_output,
    child_memory,
    agent_instance,
    max_rounds=3,
):
    """Inner loop: Planner and Socrates discuss until Socrates approves.

    Faithfully mirrors the reference discussion_until_approval():
    - PI (Socrates) uses multi-turn conversation (pi_messages persists and
      accumulates across rounds AND across calls)
    - Planner responds to questions (fresh context per call, like scientist
      resetting per session)
    - Loop until [APPROVED] or max_rounds reached

    Args:
        pi_messages: Socrates conversation history (mutated in-place,
            persists across calls via SocratesState).
        plan_text: Initial plan text to review.
        task_desc: Task description.
        parent_output: Previous execution output.
        child_memory: Memory of previous sibling attempts.
        agent_instance: AgentSearch instance for LLM config.
        max_rounds: Max discussion rounds before forcing through.

    Returns:
        (final_plan_text, approved, rounds_used)
    """
    current_plan = plan_text
    planner_response = ""

    for round_num in range(max_rounds):
        logger.info(f"[Socrates] Discussion round {round_num + 1}/{max_rounds}")

        # --- Socrates reviews (multi-turn: sees full conversation history) ---
        if round_num == 0:
            user_msg = _pi_initial_review_prompt(
                current_plan, task_desc, parent_output, child_memory,
            )
        else:
            user_msg = _pi_followup_review_prompt(planner_response)

        pi_messages.append({"role": "user", "content": user_msg})

        socrates_response = llm_chat(
            messages=pi_messages,
            system_message=SOCRATES_A_SYSTEM,
            model=agent_instance.acfg.feedback.model,
            temperature=agent_instance.acfg.feedback.temp,
            cfg=agent_instance.cfg,
        )

        if not isinstance(socrates_response, str):
            socrates_response = str(socrates_response)

        pi_messages.append({"role": "assistant", "content": socrates_response})

        # Check for approval
        if "[APPROVED]" in socrates_response.upper():
            logger.info(
                f"[Socrates] Plan APPROVED after {round_num + 1} round(s)"
            )
            return current_plan, True, round_num + 1

        logger.info(
            f"[Socrates] Round {round_num + 1}: questions raised, "
            f"planner responding"
        )

        # --- Planner defends/revises (fresh context, like scientist reset) ---
        planner_user_msg = _scientist_respond_prompt(
            socrates_response, current_plan, task_desc,
        )

        planner_response = llm_chat(
            messages=[{"role": "user", "content": planner_user_msg}],
            system_message=PLANNER_RESPOND_SYSTEM,
            model=agent_instance.acfg.code.model,
            temperature=agent_instance.acfg.code.temp,
            cfg=agent_instance.cfg,
        )

        if not isinstance(planner_response, str):
            planner_response = str(planner_response)

        # Update current plan with revised version
        if planner_response and len(planner_response.strip()) > 20:
            current_plan = planner_response.strip()

    logger.warning(
        f"[Socrates] Max rounds ({max_rounds}) reached without "
        f"approval, proceeding anyway"
    )
    return current_plan, False, max_rounds


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def review_plan(
    agent_instance,
    plan_text,
    task_desc,
    data_preview,
    parent_output,
    child_memory,
    max_rounds=3,
    socrates_state=None,
):
    """Socrates review loop for improvement plans.

    Faithful to the reference Socrates-EM architecture:
    - Socrates maintains persistent conversation history (via socrates_state)
    - Planner gets fresh context each call (scientist resets per session)
    - Multi-turn discussion within each review
    - [APPROVED] gate before plan proceeds to coding

    Args:
        agent_instance: AgentSearch instance.
        plan_text: The initial free-text plan to review.
        task_desc: Competition / task description.
        data_preview: Data preview string.
        parent_output: Parent node's execution output (str or list).
        child_memory: Memory of previous sibling attempts.
        max_rounds: Maximum discussion rounds before proceeding anyway.
        socrates_state: Persistent SocratesState (pass from AgentSearch).

    Returns:
        tuple: (final_plan_text: str, approved: bool, rounds_used: int)
    """
    if not plan_text or len(plan_text.strip()) < 20:
        logger.info("[Socrates] Plan too short for review, skipping")
        return plan_text, True, 0

    # Normalize parent_output
    if isinstance(parent_output, list):
        parent_output = "\n".join(str(x) for x in parent_output)
    else:
        parent_output = str(parent_output) if parent_output else ""

    # Get or create persistent Socrates state
    if socrates_state is None:
        socrates_state = SocratesState()

    pi_messages = socrates_state.pi_messages

    # Run the discussion loop (mirrors reference discussion_until_approval)
    final_plan, approved, rounds = discussion_until_approval(
        pi_messages=pi_messages,
        plan_text=plan_text,
        task_desc=task_desc,
        parent_output=parent_output,
        child_memory=child_memory,
        agent_instance=agent_instance,
        max_rounds=max_rounds,
    )

    # Update persistent state
    socrates_state.total_reviews += 1
    socrates_state.total_rounds += rounds
    if approved:
        socrates_state.total_approvals += 1

    logger.info(f"[Socrates] State: {socrates_state.summary()}")

    return final_plan, approved, rounds
