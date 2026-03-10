"""Resume an interrupted MLEvolve run from its saved journal."""

import argparse
import logging
import os
import tempfile
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from omegaconf import OmegaConf
from rich.status import Status

from config import Config, load_task_desc
from engine.agent_search import AgentSearch as Agent
from engine.executor import Interpreter
from engine.search_node import Journal, SearchNode
from utils import serialize
from utils.visualization import journal_to_string_tree
from utils.seed import set_global_seed
from engine.coldstart import build_guidance_description
from utils.logging_config import setup_logging
import torch


def safe_save_run(cfg, journal):
    """Atomic save: write to temp file, then rename to avoid truncation on crash."""
    from engine.search_node import filter_journal
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Save journal atomically
    for filename, obj in [
        ("journal.json", journal),
        ("filtered_journal.json", filter_journal(journal)),
    ]:
        target = cfg.log_dir / filename
        fd, tmp_path = tempfile.mkstemp(dir=cfg.log_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(serialize.dumps_json(obj))
            os.replace(tmp_path, target)
        except Exception:
            os.unlink(tmp_path)
            raise

    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")

    best_node = journal.get_best_node()
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)


def fix_deserialized_nodes(journal: Journal):
    """Fix nodes loaded from JSON: reinitialize locks, fix OMITTED fields, reset in-flight state."""
    for node in journal.nodes:
        if not isinstance(getattr(node, 'child_count_lock', None), type(threading.Lock())):
            node.child_count_lock = threading.Lock()
        # filtered_journal sets these to "<OMITTED>" string; restore to proper types
        if isinstance(node._term_out, str):
            node._term_out = [node._term_out] if node._term_out != "<OMITTED>" else []
        if isinstance(node.exc_stack, str):
            node.exc_stack = None
        # Reset in-flight state: no tasks are running after a crash
        node.lock = False
        node.expected_child_count = len(node.children)


def rebuild_agent_state(agent: Agent, journal: Journal):
    """Rebuild AgentSearch internal state from a loaded journal."""
    fix_deserialized_nodes(journal)

    agent.journal = journal
    agent.virtual_root = journal.nodes[0]
    agent.current_step = len(journal)

    # Rebuild branch tracking
    agent.branch_all_nodes = {}
    agent.branch_successful_nodes = {}
    agent.branch_node_count = {}
    max_branch_id = 0

    for node in journal.nodes[1:]:
        bid = node.branch_id
        if bid is None:
            continue
        max_branch_id = max(max_branch_id, bid)
        agent.branch_all_nodes.setdefault(bid, []).append(node)
        agent.branch_node_count[bid] = agent.branch_node_count.get(bid, 0) + 1
        if not node.is_buggy and node.metric and node.metric.value is not None:
            agent.branch_successful_nodes.setdefault(bid, []).append(node)

    agent.next_branch_id = max_branch_id + 1

    # Rebuild metric direction
    for node in journal.nodes[1:]:
        if node.metric and node.metric.value is not None and hasattr(node.metric, 'maximize'):
            agent.metric_maximize = node.metric.maximize
            break

    # Rebuild best node
    agent.best_node = None
    agent.best_metric = None
    for node in journal.nodes[1:]:
        if node.is_buggy or not node.metric or node.metric.value is None:
            continue
        if agent.best_node is None or agent.best_node.metric < node.metric:
            agent.best_node = node
            agent.best_metric = node.metric.value

    # Rebuild top candidates
    agent.top_candidates = []
    for node in journal.nodes[1:]:
        if not node.is_buggy and node.metric and node.metric.value is not None:
            from engine.solution_manager import update_top_candidates
            update_top_candidates(agent, node)

    logger = logging.getLogger("MLEvolve")
    logger.info(f"[resume] Rebuilt state: {len(journal)} nodes, "
                f"{len(agent.branch_all_nodes)} branches, "
                f"best={agent.best_metric}, next_branch_id={agent.next_branch_id}")


def resume():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to the existing run directory")
    args, remaining = parser.parse_known_args()

    run_dir = Path(args.run_dir).resolve()
    log_dir = run_dir / "logs"
    workspace_dir = run_dir / "workspace"
    config_path = log_dir / "config.yaml"
    journal_path = log_dir / "journal.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not journal_path.exists():
        raise FileNotFoundError(f"Journal not found: {journal_path}")

    # Load config from saved run (reuse dirs as-is)
    cfg = OmegaConf.load(config_path)
    cfg_schema = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)
    cfg.log_dir = log_dir
    cfg.workspace_dir = workspace_dir

    if cfg.torch_hub_dir:
        torch.hub.set_dir(cfg.torch_hub_dir)
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    logger.info(f'Resuming run "{cfg.exp_name}" from {run_dir}')

    task_desc = load_task_desc(cfg)

    if cfg.coldstart.use_coldstart:
        cfg.coldstart.description = build_guidance_description(cfg)

    # Load existing journal
    loaded_journal = serialize.load_json(journal_path, Journal)
    completed = len(loaded_journal) - 1  # exclude virtual root
    total_steps = cfg.agent.steps
    logger.info(f"[resume] Loaded journal: {len(loaded_journal)} nodes, {completed}/{total_steps} steps completed")

    if completed >= total_steps:
        logger.info(f"[resume] Run already complete ({completed}/{total_steps}). Nothing to do.")
        return

    # Create agent with empty journal (then patch it)
    empty_journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=empty_journal,
    )

    rebuild_agent_state(agent, loaded_journal)
    agent.update_data_preview()
    agent.search_start_time = time.time()

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg
    )

    status = Status("[green]Generating code...")

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def step_task(node=None):
        if node:
            logger.info(f"[step_task] Processing node: {node.id}")
        else:
            logger.info(f"[step_task] Processing virtual root node.")
        return agent.step(exec_callback=exec_callback, node=node)

    max_workers = interpreter.max_parallel_run
    lock = threading.Lock()

    logger.info(f"[resume] Starting from step {completed}/{total_steps}, max_workers={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()

        initial_tasks = min(max_workers, total_steps - completed)
        for _ in range(initial_tasks):
            futures.add(executor.submit(step_task))
            logger.info(f"📤 Submitted step_task to fill thread pool")

        while completed < total_steps:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            for fut in done:
                futures.remove(fut)
                try:
                    cur_node = fut.result()
                    if cur_node:
                        logger.info(f"✅ Task completed: node_id={cur_node.id}, step={cur_node.step}, "
                                    f"is_buggy={cur_node.is_buggy}, "
                                    f"metric={cur_node.metric.value if cur_node.metric else 'N/A'}")
                    else:
                        logger.warning(f"⚠️  Task returned None")
                except Exception as e:
                    logger.exception(f"❌ Exception during task execution: {e}")
                    cur_node = None

                with lock:
                    safe_save_run(cfg, agent.journal)
                    completed = len(agent.journal) - 1
                    if completed == total_steps:
                        logger.info(journal_to_string_tree(agent.journal))

                if completed + len(futures) < total_steps:
                    futures.add(executor.submit(step_task, cur_node))
                    logger.info(f"📤 Submitted next task based on node {cur_node.id if cur_node else 'None'}")
                logger.info(f"📊 Progress: {completed}/{total_steps} steps completed, {len(futures)} tasks running")

    interpreter.cleanup_session(-1)
    logger.info(f"[resume] Run complete: {completed}/{total_steps} steps")


if __name__ == "__main__":
    resume()
