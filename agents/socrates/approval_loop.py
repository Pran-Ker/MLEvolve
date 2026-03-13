"""Synchronous discussion loop: Socrates (PI) questions a plan until approval."""

import json
import logging
import time
from pathlib import Path

from llm import chat as llm_chat, agentic_chat
from .prompts import (
    get_socrates_a_prompt,
    get_pi_initial_review_prompt,
    get_pi_followup_review_prompt,
    get_scientist_respond_to_pi_prompt,
)

logger = logging.getLogger("MLEvolve")


class SocratesState:
    """Track review statistics across the search tree."""

    def __init__(self):
        self.total_reviews = 0
        self.total_approvals = 0
        self.total_rounds = 0


ANALYZE_ATTEMPTS_TOOL = {
    "name": "analyze_past_attempts",
    "description": (
        "Search experiment memory for past improvement attempts. "
        "Returns matching records with plans, scores, and outcomes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g. 'feature engineering', 'ensemble methods')",
            },
            "include_failures": {
                "type": "boolean",
                "description": "Include failed attempts. Default true.",
            },
        },
        "required": ["query"],
    },
}


def _execute_tool(name, input_data, global_memory):
    """Execute the analyze_past_attempts tool against global memory."""
    if name != "analyze_past_attempts":
        return f"Unknown tool: {name}"
    if global_memory is None or not global_memory.records:
        return "No experiment memory available."

    query = input_data.get("query", "")
    include_failures = input_data.get("include_failures", True)
    results = global_memory.retrieve_similar_records(query_text=query, top_k=5, alpha=0.5)
    if not results:
        return f"No matching records for: '{query}'"

    formatted = []
    for i, (record, score) in enumerate(results, 1):
        if not include_failures and record.label == -1:
            continue
        meta = global_memory.node_metadata_map.get(record.record_id, {})
        label_str = {1: "SUCCESS", 0: "NEUTRAL", -1: "FAILURE"}.get(record.label, "UNKNOWN")
        entry = f"### Attempt #{i} [{label_str}]\n"
        entry += f"**Plan:** {record.description}\n"
        entry += f"**Method:** {record.method}\n"
        pm = meta.get("parent_metric")
        cm = meta.get("current_metric")
        if pm is not None and cm is not None:
            entry += f"**Score:** {pm} -> {cm}\n"
        elif cm is not None:
            entry += f"**Score:** {cm}\n"
        formatted.append(entry)

    if not formatted:
        return "No matching records after filtering."
    return "\n".join(formatted)


def _save_transcript(agent_instance, original_plan, transcript, approved, rounds):
    """Save review transcript to JSONL for the dashboard."""
    log_dir = getattr(agent_instance.cfg, 'log_dir', None)
    if not log_dir:
        return
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "original_plan": original_plan[:500],
        "approved": approved,
        "rounds": rounds,
        "transcript": transcript,
    }
    with open(log_dir / "socrates_transcripts.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def discussion_until_approval(plan_text, task_desc, agent_prompt_context,
                              stage_name, agent_instance, max_rounds=3):
    """Socrates questions the plan, the calling agent defends/revises, until approval.

    Args:
        plan_text: The plan to review.
        task_desc: Task description.
        agent_prompt_context: Full prompt context from the calling agent (e.g. improve
            agent's prompt_complete). Used as context when the agent defends its plan.
        stage_name: Which agent is being reviewed ("improve", "evolution", "fusion", "draft").
        agent_instance: For config, model, and memory access.
        max_rounds: Max discussion rounds before proceeding anyway.

    Returns: (final_plan, approved, rounds_used)
    """
    current_plan = plan_text
    planner_response = ""
    socrates_messages = []
    transcript = []

    # Set up memory tool if available
    global_memory = getattr(agent_instance, 'global_memory', None)
    has_memory = global_memory is not None and len(global_memory.records) > 0
    tools = [ANALYZE_ATTEMPTS_TOOL] if has_memory else None
    tool_executor = (
        lambda name, inp: _execute_tool(name, inp, global_memory)
    ) if has_memory else None

    for round_num in range(max_rounds):
        logger.info(f"[Socrates] Round {round_num + 1}/{max_rounds} (reviewing {stage_name} agent)")

        # --- Socrates reviews the plan ---
        if round_num == 0:
            stage_context = (
                f"You are reviewing a plan from the {stage_name} agent. "
                f"Ask questions pertinent to the {stage_name} stage.\n\n"
            )
            user_msg = stage_context + get_pi_initial_review_prompt(current_plan)
        else:
            user_msg = get_pi_followup_review_prompt(planner_response)

        socrates_messages.append({"role": "user", "content": user_msg})

        socrates_response, socrates_messages = agentic_chat(
            messages=socrates_messages,
            system_message=get_socrates_a_prompt(),
            tools=tools,
            tool_executor=tool_executor,
            cfg=agent_instance.cfg,
            model=agent_instance.acfg.feedback.model,
            temperature=agent_instance.acfg.feedback.temp,
        )

        # Check for approval
        if "[APPROVED]" in socrates_response.upper():
            logger.info(f"[Socrates] APPROVED after {round_num + 1} round(s)")
            transcript.append({"round": round_num + 1, "socrates": socrates_response})
            _save_transcript(agent_instance, plan_text, transcript, True, round_num + 1)
            return current_plan, True, round_num + 1

        # --- The calling agent defends / revises its plan ---
        planner_msg = get_scientist_respond_to_pi_prompt("Socrates", socrates_response)
        planner_response = llm_chat(
            messages=[{"role": "user", "content": (
                f"{planner_msg}\n\nYour current plan:\n{current_plan}\n\nTask: {task_desc}"
            )}],
            system_message=agent_prompt_context,
            model=agent_instance.acfg.code.model,
            temperature=agent_instance.acfg.code.temp,
            cfg=agent_instance.cfg,
        )

        if not isinstance(planner_response, str):
            planner_response = str(planner_response)

        transcript.append({
            "round": round_num + 1,
            "socrates": socrates_response,
            "planner": planner_response,
        })

        # If planner gave a substantial response, treat it as the revised plan
        if planner_response and len(planner_response.strip()) > 20:
            current_plan = planner_response.strip()

    logger.warning(f"[Socrates] Max rounds ({max_rounds}) reached, proceeding anyway")
    _save_transcript(agent_instance, plan_text, transcript, False, max_rounds)
    return current_plan, False, max_rounds


def review_plan(agent_instance, plan_text, task_desc, data_preview,
                parent_output, child_memory, agent_prompt_context="",
                stage_name="improve", max_rounds=3, socrates_state=None):
    """Entry point for Socrates plan review.

    Returns: (final_plan, approved, rounds_used)
    """
    if not plan_text or len(plan_text.strip()) < 20:
        logger.info("[Socrates] Plan too short for review, skipping")
        return plan_text, True, 0

    if isinstance(parent_output, list):
        parent_output = "\n".join(str(x) for x in parent_output)
    else:
        parent_output = str(parent_output) if parent_output else ""

    if socrates_state is None:
        socrates_state = SocratesState()

    final_plan, approved, rounds = discussion_until_approval(
        plan_text=plan_text,
        task_desc=task_desc,
        agent_prompt_context=agent_prompt_context,
        stage_name=stage_name,
        agent_instance=agent_instance,
        max_rounds=max_rounds,
    )

    socrates_state.total_reviews += 1
    socrates_state.total_rounds += rounds
    if approved:
        socrates_state.total_approvals += 1

    return final_plan, approved, rounds
