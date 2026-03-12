"""Socrates review package — recursive PI questioning for plan validation."""

import logging

from .approval_loop import review_plan, SocratesState  # noqa: F401

logger = logging.getLogger("MLEvolve")


def socratic_review(
    agent_instance,
    plan_text,
    parent_output="",
    child_memory="",
):
    """Run Socrates review if enabled. Returns (possibly revised) plan text.

    Single integration point for all agents. If use_socrates_review is False,
    returns plan_text unchanged.
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
        max_rounds=getattr(agent_instance.acfg, 'socrates_max_rounds', 3),
        socrates_state=getattr(agent_instance, 'socrates_state', None),
    )
    logger.info(f"[Socrates] Review complete: approved={approved}, rounds={rounds}")
    return final_plan
