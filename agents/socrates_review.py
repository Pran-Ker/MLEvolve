"""Socrates review loop with tool access for improvement plans.

Socrates (PI) reviews plans with on-demand access to global memory via a
sub-agent tool. Key architectural properties:

1. Local message scope — each review_plan() call gets fresh session messages,
   bounded by max_rounds. No cross-review message accumulation.
2. Within a single review, multi-turn is fine (Socrates sees its own prior
   questions + planner responses for this session).
3. Cross-review context comes from the analyze_past_attempts tool, which
   retrieves from GlobalMemoryLayer and uses an LLM sub-agent to analyze.
4. [APPROVED] gate before plan proceeds to coding.
5. Socrates is told the planner does NOT have history of past results —
   only Socrates can access that via tools.
"""

import json
import logging
import time
from pathlib import Path

from llm import chat as llm_chat, agentic_chat

logger = logging.getLogger("MLEvolve")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SOCRATES_A_SYSTEM = (
    "You are Socrates A, a PI (advisor) to a machine learning planning agent "
    "solving a Kaggle challenge.\n\n"
    "IMPORTANT CONTEXT: The planning agent does NOT have access to the history "
    "of past experiment results, scores, or previously tried approaches. Only "
    "you have that access via the analyze_past_attempts tool. The planner is "
    "working from its current context alone. Use your tool to look up what has "
    "been tried before, then question the planner about whether their proposed "
    "approach is truly different and whether they've considered the failure "
    "modes of similar past attempts.\n\n"
    "Your focus areas:\n"
    "- Statistical methodology and rigor\n"
    "- Experimental design and validation strategy\n"
    "- Feature engineering rationale\n"
    "- Model selection justification\n"
    "- Potential data leakage or overfitting risks\n"
    "- Whether the proposed improvement is meaningfully different from "
    "past attempts (use your tool to verify)\n\n"
    "Your role:\n"
    "- Use analyze_past_attempts to check what has been tried and what "
    "worked or failed before asking questions\n"
    "- Ask probing questions to help the planning agent think deeply "
    "about METHODOLOGY — ground your questions in actual past results\n"
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
# Prompt builders
# ---------------------------------------------------------------------------

def _pi_initial_review_prompt(plan_text, task_desc, parent_output, child_memory):
    """Prompt for PI's first review of the planner's report."""
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
        "First, use your analyze_past_attempts tool to check what similar "
        "approaches have been tried before and their outcomes.\n"
        "Then verify the plan is methodologically sound and proposes a "
        "meaningfully different approach from previous attempts.\n"
        "Ask 2-3 probing questions about the methodology, "
        "OR if the plan is solid, respond with [APPROVED]."
    )
    return "\n".join(parts)


def _pi_followup_review_prompt(planner_response):
    """Prompt for PI's follow-up review after planner responds."""
    return (
        "The planning agent responds:\n\n"
        "--- RESPONSE ---\n"
        f"{planner_response}\n"
        "--- END ---\n\n"
        "If satisfied with their reasoning and revised plan, respond "
        "with [APPROVED].\nOtherwise, ask follow-up questions."
    )


def _scientist_respond_prompt(pi_response, plan_text, task_desc):
    """Prompt for planner to respond to PI questions."""
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
# Sub-agent tool definition + executor
# ---------------------------------------------------------------------------

ANALYZE_ATTEMPTS_TOOL = {
    "name": "analyze_past_attempts",
    "description": (
        "Search the experiment memory for past improvement attempts. "
        "Returns matching records with their plans, code approaches, "
        "metric scores, and outcomes (success/failure), plus an LLM-powered "
        "analysis of patterns. Use this to understand what has been tried "
        "before and what worked or failed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query describing the approach or technique to "
                    "look up (e.g. 'feature engineering', 'ensemble methods', "
                    "'learning rate tuning', 'attention mechanism')"
                ),
            },
            "include_failures": {
                "type": "boolean",
                "description": "Include failed attempts in results. Default true.",
            },
        },
        "required": ["query"],
    },
}


def _execute_tool(name, input_data, global_memory, cfg):
    """Execute Socrates tool call. Sub-agent: retrieves from memory, uses LLM to analyze."""
    if name != "analyze_past_attempts":
        return f"Unknown tool: {name}"

    if global_memory is None or not global_memory.records:
        return "No memory data available. This is an early stage of the search with no prior attempts recorded."

    query = input_data.get("query", "")
    include_failures = input_data.get("include_failures", True)

    results = global_memory.retrieve_similar_records(query_text=query, top_k=5, alpha=0.5)

    if not results:
        return f"No matching records found for query: '{query}'"

    # Basic stats
    total = len(global_memory.records)
    successes = sum(1 for r in global_memory.records if r.label == 1)
    failures = sum(1 for r in global_memory.records if r.label == -1)

    formatted = [f"Memory stats: {total} total attempts ({successes} successful, {failures} failed)\n"]

    for i, (record, score) in enumerate(results, 1):
        if not include_failures and record.label == -1:
            continue

        meta = global_memory.node_metadata_map.get(record.record_id, {})
        label_str = {1: "SUCCESS", 0: "NEUTRAL", -1: "FAILURE"}.get(record.label, "UNKNOWN")

        entry = f"### Attempt #{i} [{label_str}]\n"
        entry += f"**Stage:** {record.title}\n"
        entry += f"**Plan/Approach:** {record.description}\n"
        entry += f"**Code Summary:** {record.method}\n"

        pm = meta.get("parent_metric")
        cm = meta.get("current_metric")
        if pm is not None and cm is not None:
            entry += f"**Score:** {pm} → {cm}\n"
        elif cm is not None:
            entry += f"**Score:** {cm}\n"

        formatted.append(entry)

    if len(formatted) <= 1:
        return "No matching records found after filtering."

    records_text = "\n".join(formatted)

    # Sub-agent: LLM-powered analysis of retrieved records
    analysis = llm_chat(
        messages=[{"role": "user", "content": (
            f"Analyze these past ML experiment attempts:\n\n{records_text}\n\n"
            "Provide a concise analysis:\n"
            "1. What approaches were tried and their code strategies\n"
            "2. What worked vs failed and why (based on score changes)\n"
            "3. Score trends across attempts\n"
            "4. Key insights for deciding whether a new proposal is novel"
        )}],
        system_message="You are a concise ML experiment analyst. Be brief and actionable.",
        model=cfg.agent.feedback.model,
        temperature=0.3,
        cfg=cfg,
    )

    return f"{records_text}\n---\n**Analysis:** {analysis}"


# ---------------------------------------------------------------------------
# Socrates state (tracking only, no message accumulation)
# ---------------------------------------------------------------------------

class SocratesState:
    """Tracking state for Socrates reviewer across all review calls.

    No message accumulation — each review gets fresh local messages.
    Cross-review context comes from the analyze_past_attempts tool
    querying GlobalMemoryLayer on demand.
    """

    def __init__(self):
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
# Transcript persistence
# ---------------------------------------------------------------------------

def _save_transcript(agent_instance, original_plan, transcript, approved, rounds):
    """Append a Socrates review transcript to the log directory."""
    log_dir = getattr(agent_instance.cfg, 'log_dir', None)
    if log_dir is None:
        return
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = log_dir / "socrates_transcripts.jsonl"
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "original_plan": original_plan[:500],
        "approved": approved,
        "rounds": rounds,
        "transcript": transcript,
    }
    with open(out_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"[Socrates] Transcript saved ({rounds} rounds, approved={approved})")


# ---------------------------------------------------------------------------
# Core discussion loop
# ---------------------------------------------------------------------------

def discussion_until_approval(
    plan_text,
    task_desc,
    parent_output,
    child_memory,
    agent_instance,
    max_rounds=3,
):
    """Socrates and planner discuss until [APPROVED] or max_rounds.

    Messages are local to this review call (bounded by max_rounds).
    Socrates has tool access to query global memory for past attempts.
    Planner gets fresh single-turn context each round.

    Returns:
        (final_plan_text, approved, rounds_used)
    """
    current_plan = plan_text
    planner_response = ""
    session_messages = []  # local to this review
    transcript = []  # structured Q&A log for dashboard

    # Tool setup — only if global memory has data
    global_memory = getattr(agent_instance, 'global_memory', None)
    has_memory = global_memory is not None and len(global_memory.records) > 0
    tools = [ANALYZE_ATTEMPTS_TOOL] if has_memory else None
    tool_executor = (
        lambda name, inp: _execute_tool(name, inp, global_memory, agent_instance.cfg)
    ) if has_memory else None

    for round_num in range(max_rounds):
        logger.info(f"[Socrates] Discussion round {round_num + 1}/{max_rounds}")

        # --- Socrates reviews (multi-turn within this session) ---
        if round_num == 0:
            user_msg = _pi_initial_review_prompt(
                current_plan, task_desc, parent_output, child_memory,
            )
        else:
            user_msg = _pi_followup_review_prompt(planner_response)

        session_messages.append({"role": "user", "content": user_msg})

        socrates_response, session_messages = agentic_chat(
            messages=session_messages,
            system_message=SOCRATES_A_SYSTEM,
            tools=tools,
            tool_executor=tool_executor,
            cfg=agent_instance.cfg,
            model=agent_instance.acfg.feedback.model,
            temperature=agent_instance.acfg.feedback.temp,
        )

        # Check for approval
        if "[APPROVED]" in socrates_response.upper():
            logger.info(
                f"[Socrates] Plan APPROVED after {round_num + 1} round(s)"
            )
            transcript.append({"round": round_num + 1, "socrates": socrates_response, "planner": None, "approved": True})
            _save_transcript(agent_instance, plan_text, transcript, approved=True, rounds=round_num + 1)
            return current_plan, True, round_num + 1

        logger.info(
            f"[Socrates] Round {round_num + 1}: questions raised, "
            f"planner responding"
        )

        # --- Planner defends/revises (fresh single-turn context) ---
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

        transcript.append({"round": round_num + 1, "socrates": socrates_response, "planner": planner_response, "approved": False})

        # Update current plan with revised version
        if planner_response and len(planner_response.strip()) > 20:
            current_plan = planner_response.strip()

    logger.warning(
        f"[Socrates] Max rounds ({max_rounds}) reached without "
        f"approval, proceeding anyway"
    )
    _save_transcript(agent_instance, plan_text, transcript, approved=False, rounds=max_rounds)
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

    Each call gets fresh local messages (bounded by max_rounds).
    Socrates queries global memory on-demand via tool access.
    Planner gets fresh context each round (no history access).

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

    if socrates_state is None:
        socrates_state = SocratesState()

    final_plan, approved, rounds = discussion_until_approval(
        plan_text=plan_text,
        task_desc=task_desc,
        parent_output=parent_output,
        child_memory=child_memory,
        agent_instance=agent_instance,
        max_rounds=max_rounds,
    )

    socrates_state.total_reviews += 1
    socrates_state.total_rounds += rounds
    if approved:
        socrates_state.total_approvals += 1

    logger.info(f"[Socrates] State: {socrates_state.summary()}")

    return final_plan, approved, rounds
