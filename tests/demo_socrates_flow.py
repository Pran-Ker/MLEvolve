"""Trace through exact message flows for Socrates review scenarios.

Prints every message, tool call, and response to show exactly what happens.
No API calls — uses realistic mock responses.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.memory.record import MemRecord
from agents.socrates_review import (
    SocratesState, review_plan, ANALYZE_ATTEMPTS_TOOL, _execute_tool,
)

# ---------------------------------------------------------------------------
# Shared mock config
# ---------------------------------------------------------------------------

@dataclass
class MockStageConfig:
    model: str = "claude-sonnet-4-20250514"
    temp: float = 0.7
    api_key: str = ""
    base_url: str = ""

@dataclass
class MockAgentConfig:
    code: MockStageConfig = None
    feedback: MockStageConfig = None
    use_socrates_review: bool = True
    socrates_max_rounds: int = 3
    def __post_init__(self):
        self.code = self.code or MockStageConfig()
        self.feedback = self.feedback or MockStageConfig()

@dataclass
class MockConfig:
    agent: MockAgentConfig = None
    def __post_init__(self):
        self.agent = self.agent or MockAgentConfig()


def make_agent(with_memory=True):
    agent = MagicMock()
    agent.cfg = MockConfig()
    agent.acfg = agent.cfg.agent
    if with_memory:
        agent.global_memory = make_memory()
    else:
        agent.global_memory = None
    return agent


def make_memory():
    mem = MagicMock()
    mem.records = [
        MemRecord("node_a1", "improve - a1b2c3d4", "Add target encoding for categorical features with LightGBM",
                   "LGBMClassifier(n_estimators=800, learning_rate=0.03); TargetEncoder(cols=['cat1','cat2'])", 1),
        MemRecord("node_b2", "improve - b2c3d4e5", "XGBoost with feature selection using mutual information",
                   "XGBClassifier(n_estimators=500); SelectKBest(mutual_info_classif, k=20)", -1),
        MemRecord("node_c3", "improve - c3d4e5f6", "Ensemble stacking: LightGBM + CatBoost + logistic meta-learner",
                   "StackingClassifier([LGBMClassifier(), CatBoostClassifier()], LogisticRegression())", 0),
        MemRecord("node_d4", "draft - d4e5f6g7", "Baseline random forest with default hyperparameters",
                   "RandomForestClassifier(n_estimators=100, random_state=42)", 1),
        MemRecord("node_e5", "improve - e5f6g7h8", "Neural network with entity embeddings for categoricals",
                   "nn.Sequential: EmbeddingLayer → Dense(256) → Dense(128) → Dense(1); Adam(lr=1e-3)", -1),
    ]
    mem.node_metadata_map = {
        "node_a1": {"parent_metric": 0.8451, "current_metric": 0.8623},
        "node_b2": {"parent_metric": 0.8623, "current_metric": 0.8412},
        "node_c3": {"parent_metric": 0.8623, "current_metric": 0.8619},
        "node_d4": {"current_metric": 0.8230},
        "node_e5": {"parent_metric": 0.8623, "current_metric": 0.8301},
    }
    # retrieve_similar_records will be overridden per scenario
    return mem


def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ===================================================================
# EXAMPLE 1: Socrates uses tool, finds overlap, questions planner
# ===================================================================

def example_1_tool_use_then_questions():
    sep("EXAMPLE 1: Socrates uses tool → finds similar past attempt → questions planner")

    agent = make_agent(with_memory=True)
    state = SocratesState()

    # Memory returns: target encoding was tried before (succeeded), and XGBoost failed
    agent.global_memory.retrieve_similar_records.return_value = [
        (agent.global_memory.records[0], 0.91),  # target encoding + LightGBM → SUCCESS
        (agent.global_memory.records[1], 0.84),  # XGBoost + feature selection → FAILURE
        (agent.global_memory.records[4], 0.62),  # neural net → FAILURE
    ]

    round_num = [0]

    def mock_agentic(messages, **kwargs):
        round_num[0] += 1
        tools = kwargs.get("tools")
        executor = kwargs.get("tool_executor")

        if round_num[0] == 1:
            # --- ROUND 1: Socrates calls the tool first, then asks questions ---
            print("  [Socrates sees plan, decides to check memory]")
            print(f"  → Tool available: {tools[0]['name'] if tools else 'None'}")

            # Simulate what agentic_chat does: call tool, get result, then respond
            tool_result = executor("analyze_past_attempts", {"query": "target encoding gradient boosting"})
            print(f"\n  [Tool call: analyze_past_attempts('target encoding gradient boosting')]")
            print(f"  [Tool result preview (first 500 chars):]")
            for line in tool_result[:500].split('\n'):
                print(f"    {line}")
            print(f"    ...")

            response = (
                "I checked the experiment history and have concerns:\n\n"
                "1. Target encoding was already tried with LightGBM (attempt #1) and improved "
                "the score from 0.8451 → 0.8623. Your plan proposes target encoding again but "
                "with CatBoost — how specifically does CatBoost's handling of categoricals differ "
                "from what LightGBM already achieved with target encoding?\n\n"
                "2. The last XGBoost attempt (#2) with feature selection actually WORSENED the score "
                "to 0.8412. Your plan includes feature selection — what makes you confident it won't "
                "cause the same degradation?\n\n"
                "3. The neural network attempt (#3) with entity embeddings failed badly (0.8623 → 0.8301). "
                "What evidence suggests this tabular dataset benefits from the complexity of embeddings "
                "over the simpler target encoding that already worked?"
            )
            print(f"\n  [Socrates response:]")
            for line in response.split('\n'):
                print(f"    {line}")

            return response, messages + [{"role": "assistant", "content": response}]

        elif round_num[0] == 2:
            # --- ROUND 2: Socrates reads planner's defense, approves ---
            planner_said = messages[-1]["content"]
            print(f"\n  [Socrates reads planner's defense (first 200 chars):]")
            print(f"    {planner_said[:200]}...")

            response = (
                "[APPROVED] Good differentiation. The ordered target statistics in CatBoost "
                "are genuinely different from the manual target encoding tried before, and your "
                "validation strategy addresses the overfitting risk. Proceed."
            )
            print(f"\n  [Socrates response:]")
            print(f"    {response}")

            return response, messages + [{"role": "assistant", "content": response}]

    def mock_planner_chat(messages, **kwargs):
        response = (
            "Thank you for the thorough analysis. Let me address each point:\n\n"
            "1. CatBoost uses ordered target statistics (not standard target encoding) — it "
            "applies a time-based ordering that prevents target leakage inherently, unlike the "
            "manual TargetEncoder used in attempt #1.\n\n"
            "2. I'm dropping the feature selection step entirely based on attempt #2's failure. "
            "Instead, CatBoost will use all features with its native categorical handling.\n\n"
            "3. I'm NOT using embeddings. CatBoost's categorical handling is much simpler than "
            "neural net embeddings and has strong regularization built in.\n\n"
            "Revised plan: Use CatBoost with native categorical feature handling (ordered target "
            "statistics), all features included, 5-fold stratified CV, early stopping on validation "
            "loss with patience=50."
        )
        print(f"\n  [Planner defends (fresh single-turn, no history access):]")
        for line in response.split('\n'):
            print(f"    {line}")
        return response

    with patch("agents.socrates_review.agentic_chat", side_effect=mock_agentic):
        with patch("agents.socrates_review.llm_chat", side_effect=mock_planner_chat):
            plan, approved, rounds = review_plan(
                agent_instance=agent,
                plan_text=(
                    "Use CatBoost with target encoding for categorical features. "
                    "Apply mutual information feature selection to keep top 25 features. "
                    "Use entity embeddings as an alternative representation. "
                    "Train with 5-fold CV and early stopping."
                ),
                task_desc="Binary classification on tabular dataset with 40 features (15 categorical)",
                data_preview="",
                parent_output="Current best: AUC=0.8623 (LightGBM + target encoding)",
                child_memory="Sibling tried random forest, got AUC=0.8510",
                max_rounds=3,
                socrates_state=state,
            )

    print(f"\n  --- RESULT ---")
    print(f"  Approved: {approved}")
    print(f"  Rounds: {rounds}")
    print(f"  Final plan (first 150 chars): {plan[:150]}...")
    print(f"  State: {state.summary()}")


# ===================================================================
# EXAMPLE 2: No global memory — Socrates works without tools
# ===================================================================

def example_2_no_memory():
    sep("EXAMPLE 2: No global memory — Socrates reviews without tool access")

    agent = make_agent(with_memory=False)
    state = SocratesState()

    def mock_agentic(messages, **kwargs):
        tools = kwargs.get("tools")
        print(f"  [Tools available: {tools}]")
        print(f"  [Messages count: {len(messages)}]")

        response = (
            "[APPROVED] The plan proposes a reasonable baseline with logistic regression "
            "and proper cross-validation. Good starting point."
        )
        print(f"  [Socrates response (no tool access, reviews on plan alone):]")
        print(f"    {response}")
        return response, messages + [{"role": "assistant", "content": response}]

    with patch("agents.socrates_review.agentic_chat", side_effect=mock_agentic):
        plan, approved, rounds = review_plan(
            agent_instance=agent,
            plan_text="Build baseline logistic regression with StandardScaler, 5-fold stratified CV",
            task_desc="Binary classification task",
            data_preview="",
            parent_output="",
            child_memory="",
            max_rounds=3,
            socrates_state=state,
        )

    print(f"\n  --- RESULT ---")
    print(f"  Approved: {approved}, Rounds: {rounds}")
    print(f"  State: {state.summary()}")


# ===================================================================
# EXAMPLE 3: Socrates rejects — hits max rounds
# ===================================================================

def example_3_max_rounds():
    sep("EXAMPLE 3: Socrates never satisfied — max rounds reached")

    agent = make_agent(with_memory=True)
    state = SocratesState()

    agent.global_memory.retrieve_similar_records.return_value = [
        (agent.global_memory.records[2], 0.88),  # stacking → NEUTRAL (0.8623→0.8619)
    ]

    round_num = [0]

    def mock_agentic(messages, **kwargs):
        round_num[0] += 1
        responses = {
            1: "I checked memory — stacking was tried before (attempt #1) and gave NO improvement "
               "(0.8623 → 0.8619). Your plan is essentially the same stacking approach with minor "
               "model swaps. How is this meaningfully different?",
            2: "You say you're adding a neural net to the stack, but neural nets already failed on "
               "this dataset (0.8623 → 0.8301). Adding a weak learner to a stack typically hurts. "
               "What's your evidence this will work?",
        }
        r = responses.get(round_num[0], "Still not convinced.")
        print(f"  [Round {round_num[0]} — Socrates:]")
        print(f"    {r[:200]}...")
        return r, messages + [{"role": "assistant", "content": r}]

    planner_round = [0]

    def mock_planner(messages, **kwargs):
        planner_round[0] += 1
        responses = {
            1: "I'll differentiate by adding a neural net as third base learner and using Bayesian optimization for the meta-learner hyperparameters.",
            2: "The neural net in the stack will use much simpler architecture (single hidden layer, 64 units) compared to the failed attempt.",
        }
        r = responses.get(planner_round[0], "Further revisions.")
        print(f"  [Round {planner_round[0]} — Planner defends:]")
        print(f"    {r}")
        return r

    with patch("agents.socrates_review.agentic_chat", side_effect=mock_agentic):
        with patch("agents.socrates_review.llm_chat", side_effect=mock_planner):
            plan, approved, rounds = review_plan(
                agent_instance=agent,
                plan_text="Ensemble stacking with LightGBM + CatBoost + logistic meta-learner, similar to previous but with hyperparameter tuning",
                task_desc="Binary classification on tabular dataset",
                data_preview="",
                parent_output="AUC=0.8623",
                child_memory="",
                max_rounds=2,
                socrates_state=state,
            )

    print(f"\n  --- RESULT ---")
    print(f"  Approved: {approved} (forced through after max rounds)")
    print(f"  Rounds: {rounds}")
    print(f"  State: {state.summary()}")


# ===================================================================
# EXAMPLE 4: Two consecutive reviews — messages don't leak
# ===================================================================

def example_4_isolation():
    sep("EXAMPLE 4: Two reviews back-to-back — session messages stay isolated")

    agent = make_agent(with_memory=True)
    state = SocratesState()

    agent.global_memory.retrieve_similar_records.return_value = [
        (agent.global_memory.records[0], 0.90),
    ]

    review_num = [0]

    def mock_agentic(messages, **kwargs):
        review_num[0] += 1
        msg_count = len(messages)
        first_msg_preview = messages[0]["content"][:80] if messages else "?"

        print(f"  [Review #{review_num[0]}]")
        print(f"    Messages passed in: {msg_count}")
        print(f"    First message starts with: \"{first_msg_preview}...\"")

        if msg_count > 1:
            print(f"    *** BUG: Should be 1 message (fresh session), got {msg_count} ***")
        else:
            print(f"    Correct: fresh session, no leaked messages from prior review")

        return "[APPROVED] Proceed.", messages + [{"role": "assistant", "content": "[APPROVED]"}]

    with patch("agents.socrates_review.agentic_chat", side_effect=mock_agentic):
        print("  --- Review A ---")
        review_plan(agent, "Plan A: Add feature interactions using PolynomialFeatures degree=2",
                     "Classification task", "", "AUC=0.86", "", max_rounds=3, socrates_state=state)

        print("\n  --- Review B ---")
        review_plan(agent, "Plan B: Try gradient boosting with Bayesian hyperparameter optimization",
                     "Classification task", "", "AUC=0.86", "", max_rounds=3, socrates_state=state)

    print(f"\n  --- RESULT ---")
    print(f"  State after both: {state.summary()}")
    print(f"  Both reviews started with 1 message = sessions are isolated")


# ===================================================================
# EXAMPLE 5: Raw tool output — what Socrates actually sees
# ===================================================================

def example_5_raw_tool_output():
    sep("EXAMPLE 5: Raw tool output — what the sub-agent returns to Socrates")

    mem = make_memory()
    cfg = MockConfig()

    # Return all 5 records
    mem.retrieve_similar_records.return_value = [
        (mem.records[0], 0.93),
        (mem.records[1], 0.87),
        (mem.records[4], 0.76),
        (mem.records[2], 0.68),
        (mem.records[3], 0.51),
    ]

    mock_analysis = (
        "Key patterns:\n"
        "- Target encoding + LightGBM was the only successful improvement (+1.7% AUC)\n"
        "- Feature selection consistently hurts (XGBoost attempt lost 2.1% AUC)\n"
        "- Neural networks are a poor fit for this tabular data (-3.2% AUC)\n"
        "- Stacking provided negligible benefit (-0.04% AUC, not worth complexity)\n"
        "- Best strategy: stick with gradient boosting, focus on encoding not selection"
    )

    with patch("agents.socrates_review.llm_chat", return_value=mock_analysis):
        result = _execute_tool(
            "analyze_past_attempts",
            {"query": "gradient boosting categorical features"},
            mem, cfg,
        )

    print("  [Full tool output that Socrates receives:]")
    print("  " + "-"*60)
    for line in result.split('\n'):
        print(f"  {line}")
    print("  " + "-"*60)


# ===================================================================
# Run all examples
# ===================================================================

if __name__ == "__main__":
    example_1_tool_use_then_questions()
    example_2_no_memory()
    example_3_max_rounds()
    example_4_isolation()
    example_5_raw_tool_output()

    print(f"\n{'='*70}")
    print("  All 5 examples completed.")
    print(f"{'='*70}")
