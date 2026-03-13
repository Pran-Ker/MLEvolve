"""Socrates review package — recursive PI questioning for plan validation."""

import logging

from .approval_loop import review_plan, SocratesState  # noqa: F401

logger = logging.getLogger("MLEvolve")


def socratic_review(
    agent_instance,
    plan_text,
    parent_output="",
    child_memory="",
    agent_prompt_context="",
    stage_name="improve",
):
    """Run Socrates review if enabled. Returns (possibly revised) plan text.

    Args:
        agent_instance: The agent whose plan is being reviewed.
        plan_text: The plan text to review.
        parent_output: Previous execution output.
        child_memory: Memory from parent node.
        agent_prompt_context: Full prompt context from the calling agent.
            Used as the system message when the agent defends its plan.
        stage_name: Which agent is being reviewed ("improve", "evolution", etc.).
    """
    if not getattr(agent_instance.acfg, 'use_socrates_review', False):
        return plan_text

    final_plan, approved, rounds = review_plan(
        agent_instance=agent_instance,
        plan_text=plan_text,
        task_desc=agent_instance.task_desc,
        data_preview=getattr(agent_instance, 'data_preview', ''),
        parent_output=parent_output,
        child_memory=child_memory,
        agent_prompt_context=agent_prompt_context,
        stage_name=stage_name,
        max_rounds=getattr(agent_instance.acfg, 'socrates_max_rounds', 3),
        socrates_state=getattr(agent_instance, 'socrates_state', None),
    )
    logger.info(f"[Socrates] Review complete: approved={approved}, rounds={rounds}")
    return final_plan


def review_planning_result(
    agent_instance,
    planning_result,
    parent_output="",
    child_memory="",
    agent_prompt_context="",
    stage_name="improve",
):
    """Review a structured planning result (from run_planner) via Socrates.

    Formats the planning_result dict into readable text, reviews it,
    and updates the 'reason' field with the revised text.
    Returns the (possibly updated) planning_result dict.
    """
    parts = []
    reason = planning_result.get('reason', '')
    if reason:
        parts.append(f"Reasoning: {reason}")
    for mod, plan in planning_result.get('plan', {}).items():
        parts.append(f"Module '{mod}': {plan}")
    if not parts:
        return planning_result

    plan_text = "\n\n".join(parts)
    revised = socratic_review(
        agent_instance, plan_text,
        parent_output=parent_output,
        child_memory=child_memory,
        agent_prompt_context=agent_prompt_context,
        stage_name=stage_name,
    )
    if revised != plan_text:
        planning_result['reason'] = revised
    return planning_result
