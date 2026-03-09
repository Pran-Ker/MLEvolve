"""Tests for the agentic Socrates review loop.

Mocks all LLM calls — no API keys needed. Tests:
- Tool schema validity
- Sub-agent tool execution with mock global memory
- Graceful degradation (no memory)
- Discussion loop: immediate approval, multi-round, max rounds
- Tool-calling flow through agentic_chat
- Session messages stay local (no cross-review leakage)
- SocratesState tracking
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.memory.record import MemRecord
from agents.socrates_review import (
    SocratesState,
    review_plan,
    discussion_until_approval,
    ANALYZE_ATTEMPTS_TOOL,
    _execute_tool,
)


# ---------------------------------------------------------------------------
# Mock fixtures
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


def make_mock_agent(with_memory=True):
    """Create a mock AgentSearch-like object."""
    agent = MagicMock()
    agent.cfg = MockConfig()
    agent.acfg = agent.cfg.agent

    if with_memory:
        agent.global_memory = make_mock_global_memory()
    else:
        agent.global_memory = None
    return agent


def make_mock_global_memory():
    """Create a mock GlobalMemoryLayer with sample records."""
    mem = MagicMock()
    mem.records = [
        MemRecord("node_aaa", "improve - aaa11111", "Add gradient boosting with target encoding",
                   "XGBClassifier(n_estimators=500, learning_rate=0.05)", 1, "2024-01-01T00:00:00"),
        MemRecord("node_bbb", "improve - bbb22222", "Try random forest with feature selection",
                   "RandomForestClassifier(n_estimators=300, max_depth=10)", -1, "2024-01-02T00:00:00"),
        MemRecord("node_ccc", "draft - ccc33333", "Baseline logistic regression",
                   "LogisticRegression(C=1.0, solver='lbfgs')", 1, "2024-01-03T00:00:00"),
        MemRecord("node_ddd", "improve - ddd44444", "Ensemble stacking with LightGBM + XGBoost",
                   "StackingClassifier([LGBMClassifier(), XGBClassifier()])", 0, "2024-01-04T00:00:00"),
    ]
    mem.node_metadata_map = {
        "node_aaa": {"parent_metric": 0.845, "current_metric": 0.862, "exec_time": 120},
        "node_bbb": {"parent_metric": 0.862, "current_metric": 0.841, "exec_time": 95},
        "node_ccc": {"current_metric": 0.823, "exec_time": 30},
        "node_ddd": {"parent_metric": 0.862, "current_metric": 0.862, "exec_time": 200},
    }
    mem.retrieve_similar_records = MagicMock(
        return_value=[(mem.records[0], 0.92), (mem.records[1], 0.85), (mem.records[3], 0.71)]
    )
    return mem


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tool_schema():
    """Tool definition has valid JSON schema structure."""
    tool = ANALYZE_ATTEMPTS_TOOL
    assert tool["name"] == "analyze_past_attempts"
    assert "input_schema" in tool
    schema = tool["input_schema"]
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert schema["properties"]["query"]["type"] == "string"
    assert "required" in schema
    assert "query" in schema["required"]
    print("PASS: tool_schema")


def test_execute_tool_with_memory():
    """Tool returns formatted records + LLM analysis."""
    mem = make_mock_global_memory()
    cfg = MockConfig()

    with patch("agents.socrates_review.llm_chat", return_value="Analysis: XGBoost worked well, RF failed."):
        result = _execute_tool("analyze_past_attempts", {"query": "gradient boosting"}, mem, cfg)

    assert "Memory stats: 4 total attempts" in result
    assert "2 successful" in result
    assert "1 failed" in result
    assert "Attempt #1 [SUCCESS]" in result
    assert "0.845 → 0.862" in result
    assert "Attempt #2 [FAILURE]" in result
    assert "0.862 → 0.841" in result
    assert "XGBClassifier" in result
    assert "Analysis:" in result
    mem.retrieve_similar_records.assert_called_once_with(query_text="gradient boosting", top_k=5, alpha=0.5)
    print("PASS: execute_tool_with_memory")


def test_execute_tool_exclude_failures():
    """Tool filters out failures when include_failures=False."""
    mem = make_mock_global_memory()
    cfg = MockConfig()

    with patch("agents.socrates_review.llm_chat", return_value="Only successes shown."):
        result = _execute_tool("analyze_past_attempts", {"query": "boost", "include_failures": False}, mem, cfg)

    assert "[FAILURE]" not in result
    assert "[SUCCESS]" in result
    print("PASS: execute_tool_exclude_failures")


def test_execute_tool_no_memory():
    """Tool degrades gracefully with no memory."""
    result = _execute_tool("analyze_past_attempts", {"query": "anything"}, None, MockConfig())
    assert "No memory data" in result

    empty_mem = MagicMock()
    empty_mem.records = []
    result = _execute_tool("analyze_past_attempts", {"query": "anything"}, empty_mem, MockConfig())
    assert "No memory data" in result
    print("PASS: execute_tool_no_memory")


def test_execute_tool_no_results():
    """Tool handles zero retrieval results."""
    mem = make_mock_global_memory()
    mem.retrieve_similar_records.return_value = []
    cfg = MockConfig()

    result = _execute_tool("analyze_past_attempts", {"query": "nonexistent"}, mem, cfg)
    assert "No matching records" in result
    print("PASS: execute_tool_no_results")


def test_execute_tool_unknown():
    """Unknown tool name returns error."""
    result = _execute_tool("unknown_tool", {}, None, MockConfig())
    assert "Unknown tool" in result
    print("PASS: execute_tool_unknown")


def test_approval_round_1():
    """Socrates approves immediately on round 1."""
    agent = make_mock_agent(with_memory=False)

    with patch("agents.socrates_review.agentic_chat") as mock_agentic:
        mock_agentic.return_value = (
            "[APPROVED] Solid methodology, proceed.",
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "[APPROVED] Solid."}],
        )
        plan, approved, rounds = discussion_until_approval(
            plan_text="Use gradient boosting with cross-validation",
            task_desc="Predict house prices",
            parent_output="RMSE: 0.15",
            child_memory="",
            agent_instance=agent,
            max_rounds=3,
        )

    assert approved is True
    assert rounds == 1
    assert plan == "Use gradient boosting with cross-validation"
    mock_agentic.assert_called_once()
    print("PASS: approval_round_1")


def test_approval_after_discussion():
    """Socrates asks questions, planner responds, then approved on round 2."""
    agent = make_mock_agent(with_memory=False)

    call_count = [0]
    def mock_agentic_side_effect(messages, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return (
                "Why did you choose gradient boosting over neural nets?",
                messages + [{"role": "assistant", "content": "Why gradient boosting?"}],
            )
        else:
            return (
                "[APPROVED] Good justification.",
                messages + [{"role": "assistant", "content": "[APPROVED]"}],
            )

    with patch("agents.socrates_review.agentic_chat", side_effect=mock_agentic_side_effect):
        with patch("agents.socrates_review.llm_chat", return_value="Because tabular data works better with GBMs. Revised plan: ..."):
            plan, approved, rounds = discussion_until_approval(
                plan_text="Use gradient boosting with cross-validation",
                task_desc="Predict house prices",
                parent_output="RMSE: 0.15",
                child_memory="",
                agent_instance=agent,
                max_rounds=3,
            )

    assert approved is True
    assert rounds == 2
    print("PASS: approval_after_discussion")


def test_max_rounds_no_approval():
    """Socrates never approves — hits max_rounds."""
    agent = make_mock_agent(with_memory=False)

    def always_question(messages, **kwargs):
        return (
            "I have more concerns about your validation strategy.",
            messages + [{"role": "assistant", "content": "More concerns."}],
        )

    with patch("agents.socrates_review.agentic_chat", side_effect=always_question):
        with patch("agents.socrates_review.llm_chat", return_value="Here is my revised plan with better validation."):
            plan, approved, rounds = discussion_until_approval(
                plan_text="Use gradient boosting",
                task_desc="Predict house prices",
                parent_output="",
                child_memory="",
                agent_instance=agent,
                max_rounds=2,
            )

    assert approved is False
    assert rounds == 2
    assert len(plan) > 20  # planner's response became current_plan
    print("PASS: max_rounds_no_approval")


def test_tools_provided_when_memory_exists():
    """Tools are passed to agentic_chat when global memory has records."""
    agent = make_mock_agent(with_memory=True)

    with patch("agents.socrates_review.agentic_chat") as mock_agentic:
        mock_agentic.return_value = ("[APPROVED] Looks good.", [])
        discussion_until_approval(
            plan_text="Some plan here",
            task_desc="Task",
            parent_output="",
            child_memory="",
            agent_instance=agent,
            max_rounds=1,
        )

    call_kwargs = mock_agentic.call_args
    assert call_kwargs.kwargs.get("tools") == [ANALYZE_ATTEMPTS_TOOL]
    assert call_kwargs.kwargs.get("tool_executor") is not None
    print("PASS: tools_provided_when_memory_exists")


def test_no_tools_when_no_memory():
    """Tools are None when no global memory."""
    agent = make_mock_agent(with_memory=False)

    with patch("agents.socrates_review.agentic_chat") as mock_agentic:
        mock_agentic.return_value = ("[APPROVED] Fine.", [])
        discussion_until_approval(
            plan_text="Some plan here",
            task_desc="Task",
            parent_output="",
            child_memory="",
            agent_instance=agent,
            max_rounds=1,
        )

    call_kwargs = mock_agentic.call_args
    assert call_kwargs.kwargs.get("tools") is None
    assert call_kwargs.kwargs.get("tool_executor") is None
    print("PASS: no_tools_when_no_memory")


def test_tool_executor_wired_correctly():
    """The lambda tool_executor actually calls _execute_tool with the right memory."""
    agent = make_mock_agent(with_memory=True)
    captured_executor = [None]

    def capture_agentic(messages, **kwargs):
        captured_executor[0] = kwargs.get("tool_executor")
        return ("[APPROVED]", [])

    with patch("agents.socrates_review.agentic_chat", side_effect=capture_agentic):
        discussion_until_approval(
            plan_text="Plan text",
            task_desc="Task",
            parent_output="",
            child_memory="",
            agent_instance=agent,
            max_rounds=1,
        )

    executor = captured_executor[0]
    assert executor is not None

    # Call the executor directly — it should hit global_memory
    with patch("agents.socrates_review.llm_chat", return_value="Mock analysis"):
        result = executor("analyze_past_attempts", {"query": "test"})
    assert "Memory stats" in result
    assert "Attempt #1" in result
    agent.global_memory.retrieve_similar_records.assert_called()
    print("PASS: tool_executor_wired_correctly")


def test_session_messages_local():
    """Session messages don't leak between review_plan() calls."""
    agent = make_mock_agent(with_memory=False)
    state = SocratesState()

    messages_seen = []

    def track_messages(messages, **kwargs):
        messages_seen.append(len(messages))
        return ("[APPROVED]", messages + [{"role": "assistant", "content": "[APPROVED]"}])

    with patch("agents.socrates_review.agentic_chat", side_effect=track_messages):
        # First review
        review_plan(agent, "Plan A: do gradient boosting", "Task", "data", "output", "", max_rounds=3, socrates_state=state)
        # Second review
        review_plan(agent, "Plan B: try neural network approach", "Task", "data", "output", "", max_rounds=3, socrates_state=state)

    # Both calls should start with 1 message (just the initial prompt), not accumulating
    assert messages_seen[0] == 1, f"First review started with {messages_seen[0]} messages, expected 1"
    assert messages_seen[1] == 1, f"Second review started with {messages_seen[1]} messages, expected 1"
    print("PASS: session_messages_local")


def test_state_tracking():
    """SocratesState counters update correctly across reviews."""
    agent = make_mock_agent(with_memory=False)
    state = SocratesState()

    call_count = [0]
    def alternating(messages, **kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 1:
            return ("[APPROVED]", [])
        else:
            return ("Questions about your plan.", messages + [{"role": "assistant", "content": "Q"}])

    with patch("agents.socrates_review.agentic_chat", side_effect=alternating):
        with patch("agents.socrates_review.llm_chat", return_value="Revised plan with better approach here."):
            # Review 1: approved round 1 (call_count=1, odd → approved)
            review_plan(agent, "Plan A long enough to pass check", "Task", "", "", "", max_rounds=3, socrates_state=state)
            # Review 2: not approved round 1 (call_count=2, even), approved round 2 (call_count=3, odd)
            review_plan(agent, "Plan B long enough to pass check", "Task", "", "", "", max_rounds=3, socrates_state=state)

    assert state.total_reviews == 2
    assert state.total_approvals == 2
    assert state.total_rounds == 3  # 1 + 2
    print("PASS: state_tracking")


def test_short_plan_skipped():
    """Plans shorter than 20 chars skip review."""
    agent = make_mock_agent(with_memory=False)
    plan, approved, rounds = review_plan(agent, "short", "Task", "", "", "")
    assert approved is True
    assert rounds == 0
    assert plan == "short"
    print("PASS: short_plan_skipped")


def test_plan_updated_from_planner():
    """Current plan updates when planner gives substantive response."""
    agent = make_mock_agent(with_memory=False)

    call_count = [0]
    def socrates_flow(messages, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return ("What about overfitting?", messages + [{"role": "assistant", "content": "Q"}])
        return ("[APPROVED]", messages + [{"role": "assistant", "content": "[APPROVED]"}])

    revised = "REVISED: Use gradient boosting with 5-fold CV and early stopping to prevent overfitting."
    with patch("agents.socrates_review.agentic_chat", side_effect=socrates_flow):
        with patch("agents.socrates_review.llm_chat", return_value=revised):
            plan, approved, rounds = review_plan(
                agent, "Original plan: gradient boosting", "Task", "", "", "",
                max_rounds=3,
            )

    assert plan == revised
    assert approved is True
    print("PASS: plan_updated_from_planner")


def test_parent_output_list_normalized():
    """List parent_output gets joined into string."""
    agent = make_mock_agent(with_memory=False)

    with patch("agents.socrates_review.agentic_chat") as mock:
        mock.return_value = ("[APPROVED]", [])
        review_plan(agent, "Plan that is long enough here", "Task", "",
                     ["line1", "line2", "line3"], "", max_rounds=1)

    # Check the initial prompt included joined output
    call_args = mock.call_args
    messages = call_args.kwargs.get("messages") or call_args[0][0]
    first_msg = messages[0]["content"]
    assert "line1\nline2\nline3" in first_msg
    print("PASS: parent_output_list_normalized")


def test_agentic_chat_no_tools():
    """agentic_chat works as plain chat when tools=None."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="Just a text response.")]
    mock_response.stop_reason = "end_turn"

    with patch("llm.claude._setup_claude_client"):
        with patch("llm.claude._client") as mock_client:
            mock_client.messages.create.return_value = mock_response

            from llm.claude import agentic_chat
            text, msgs = agentic_chat(
                messages=[{"role": "user", "content": "Hello"}],
                cfg=MockConfig(),
                tools=None,
            )

    assert text == "Just a text response."
    assert len(msgs) == 2  # original user + assistant reply
    print("PASS: agentic_chat_no_tools")


def test_agentic_chat_tool_loop():
    """agentic_chat executes tool calls and loops back."""
    # First response: tool_use (name/input are special on MagicMock, set after init)
    tool_use_block = MagicMock(type="tool_use", id="tool_123")
    tool_use_block.name = "analyze_past_attempts"
    tool_use_block.input = {"query": "boosting"}
    text_block_1 = MagicMock(type="text", text="Let me check. ")
    resp1 = MagicMock()
    resp1.content = [text_block_1, tool_use_block]
    resp1.stop_reason = "tool_use"

    # Second response: text only
    text_block_2 = MagicMock(type="text", text="Based on past results, I have questions.")
    resp2 = MagicMock()
    resp2.content = [text_block_2]
    resp2.stop_reason = "end_turn"

    call_count = [0]
    def mock_create(**kwargs):
        call_count[0] += 1
        return resp1 if call_count[0] == 1 else resp2

    executor_calls = []
    def mock_executor(name, inp):
        executor_calls.append((name, inp))
        return "Tool result: 3 past attempts found."

    with patch("llm.claude._setup_claude_client"):
        with patch("llm.claude._client") as mock_client:
            mock_client.messages.create.side_effect = mock_create

            from llm.claude import agentic_chat
            text, msgs = agentic_chat(
                messages=[{"role": "user", "content": "Review this plan"}],
                cfg=MockConfig(),
                tools=[ANALYZE_ATTEMPTS_TOOL],
                tool_executor=mock_executor,
            )

    assert text == "Based on past results, I have questions."
    assert call_count[0] == 2  # two API calls
    assert len(executor_calls) == 1
    assert executor_calls[0] == ("analyze_past_attempts", {"query": "boosting"})
    # Messages: user + assistant(tool_use) + user(tool_result) + assistant(text)
    assert len(msgs) == 4
    assert msgs[2]["role"] == "user"  # tool_result message
    print("PASS: agentic_chat_tool_loop")


def test_agentic_chat_max_rounds():
    """agentic_chat stops after max_tool_rounds even if model keeps requesting tools."""
    tool_block = MagicMock(type="tool_use", id="t1")
    tool_block.name = "analyze_past_attempts"
    tool_block.input = {"query": "x"}
    resp = MagicMock()
    resp.content = [MagicMock(type="text", text=""), tool_block]
    resp.stop_reason = "tool_use"

    with patch("llm.claude._setup_claude_client"):
        with patch("llm.claude._client") as mock_client:
            mock_client.messages.create.return_value = resp

            from llm.claude import agentic_chat
            text, msgs = agentic_chat(
                messages=[{"role": "user", "content": "Test"}],
                cfg=MockConfig(),
                tools=[ANALYZE_ATTEMPTS_TOOL],
                tool_executor=lambda n, i: "result",
                max_tool_rounds=3,
            )

    assert mock_client.messages.create.call_count == 3
    print("PASS: agentic_chat_max_rounds")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_tool_schema,
        test_execute_tool_with_memory,
        test_execute_tool_exclude_failures,
        test_execute_tool_no_memory,
        test_execute_tool_no_results,
        test_execute_tool_unknown,
        test_approval_round_1,
        test_approval_after_discussion,
        test_max_rounds_no_approval,
        test_tools_provided_when_memory_exists,
        test_no_tools_when_no_memory,
        test_tool_executor_wired_correctly,
        test_session_messages_local,
        test_state_tracking,
        test_short_plan_skipped,
        test_plan_updated_from_planner,
        test_parent_output_list_normalized,
        test_agentic_chat_no_tools,
        test_agentic_chat_tool_loop,
        test_agentic_chat_max_rounds,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
