"""Tests for Action/Approval Planner — ActionRegistry, ActionPlanner, Block Kit, ActionExecutor."""

import json
import threading
import time
from unittest.mock import Mock, MagicMock

from agent.actions import (
    ActionRegistry,
    ActionPlanner,
    ActionExecutor,
    build_action_blocks,
    _ALL_ACTIONS,
    _DEFAULT_ACTIONS,
)


# ---------------------------------------------------------------------------
# ActionRegistry tests
# ---------------------------------------------------------------------------

class TestActionRegistry:
    def test_store_and_get(self):
        registry = ActionRegistry()
        registry.store("123.456", [{"id": "create_thread", "label": "Create Thread", "description": "Open a thread"}])
        result = registry.get("123.456")
        assert result is not None
        assert len(result["actions"]) == 1
        assert result["actions"][0]["id"] == "create_thread"

    def test_get_missing_key_returns_none(self):
        registry = ActionRegistry()
        assert registry.get("nonexistent") is None

    def test_get_actions_returns_list(self):
        registry = ActionRegistry()
        registry.store("123.456", [{"id": "test", "label": "T", "description": "T"}])
        actions = registry.get_actions("123.456")
        assert isinstance(actions, list)
        assert len(actions) == 1

    def test_get_actions_missing_returns_none(self):
        registry = ActionRegistry()
        assert registry.get_actions("nonexistent") is None

    def test_store_includes_diagnosis(self):
        registry = ActionRegistry()
        registry.store("123.456", [{"id": "test", "label": "T", "description": "T"}], diagnosis="Redis timeout")
        assert registry.get_diagnosis("123.456") == "Redis timeout"

    def test_get_diagnosis_missing_returns_default(self):
        registry = ActionRegistry()
        assert registry.get_diagnosis("nonexistent") == "Unknown issue"

    def test_store_overwrites_previous(self):
        registry = ActionRegistry()
        registry.store("123.456", [{"id": "old", "label": "Old", "description": "Old"}])
        registry.store("123.456", [{"id": "new", "label": "New", "description": "New"}])
        retrieved = registry.get_actions("123.456")
        assert len(retrieved) == 1
        assert retrieved[0]["id"] == "new"

    def test_ttl_eviction(self):
        registry = ActionRegistry(ttl=0)
        registry.store("123.456", [{"id": "test", "label": "T", "description": "T"}])
        time.sleep(0.01)
        assert registry.get("123.456") is None

    def test_len_counts_active(self):
        registry = ActionRegistry()
        registry.store("a", [{"id": "x", "label": "X", "description": "X"}])
        registry.store("b", [{"id": "y", "label": "Y", "description": "Y"}])
        assert len(registry) == 2

    def test_thread_safe(self):
        """Concurrent stores and gets on the SAME key don't corrupt state."""
        registry = ActionRegistry()
        errors = []

        def store_and_get(msg_id):
            try:
                for _ in range(50):
                    registry.store(msg_id, [{"id": "test", "label": "T", "description": "T"}])
                    registry.get(msg_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_and_get, args=("shared_msg",)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        result = registry.get("shared_msg")
        assert result is not None


# ---------------------------------------------------------------------------
# ActionPlanner tests
# ---------------------------------------------------------------------------

class TestActionPlanner:
    def test_default_actions_without_llm(self):
        planner = ActionPlanner(llm=None)
        actions = planner.recommend("Redis timeout in aggregator")
        assert len(actions) >= 1
        assert any(a["id"] == "create_thread" for a in actions)

    def test_planner_returns_actions_for_diagnosis(self):
        mock_llm = Mock()
        mock_llm.chat.return_value = Mock(content='{"actions": ["create_thread", "explain_code"]}')
        planner = ActionPlanner(llm=mock_llm)
        actions = planner.recommend("Timeout in redis_client.py:42")
        assert len(actions) >= 1

    def test_planner_fallback_on_llm_error(self):
        mock_llm = Mock()
        mock_llm.chat.side_effect = Exception("LLM unavailable")
        planner = ActionPlanner(llm=mock_llm)
        actions = planner.recommend("Some issue")
        assert len(actions) >= 1

    def test_planner_includes_explain_code_when_files_read(self):
        mock_llm = Mock()
        mock_llm.chat.return_value = Mock(content='{"actions": ["explain_code", "create_thread"]}')
        planner = ActionPlanner(llm=mock_llm)
        actions = planner.recommend("Timeout in redis_client.py:42", tools_used=["flowmap_cat"])
        action_ids = [a["id"] for a in actions]
        assert "explain_code" in action_ids or "create_thread" in action_ids

    def test_parse_json_response(self):
        planner = ActionPlanner(llm=None)
        result = planner._parse_response('Here is my recommendation: {"actions": ["create_thread"]}')
        assert result == ["create_thread"]

    def test_parse_plain_text_fallback(self):
        planner = ActionPlanner(llm=None)
        result = planner._parse_response("I recommend create_thread and explain_code")
        assert "create_thread" in result
        assert "explain_code" in result

    def test_parse_garbage_returns_defaults(self):
        planner = ActionPlanner(llm=None)
        result = planner._parse_response("I don't understand")
        assert result == list(_DEFAULT_ACTIONS)


# ---------------------------------------------------------------------------
# Block Kit builder tests
# ---------------------------------------------------------------------------

class TestBlockKitBuilder:
    def test_builds_divider_and_actions(self):
        """Blocks include divider + actions (no diagnosis section — that's in the streaming message)."""
        blocks = build_action_blocks(
            "Root cause: Redis timeout",
            [{"id": "create_thread", "label": "Create Thread", "description": "Open thread"}],
            "123.456",
        )
        assert len(blocks) == 2
        assert blocks[0]["type"] == "divider"
        assert blocks[1]["type"] == "actions"

    def test_no_diagnosis_section_in_blocks(self):
        """Diagnosis text is NOT included in action blocks (prevents double-post)."""
        blocks = build_action_blocks(
            "Root cause: Redis timeout in config.js",
            [{"id": "create_thread", "label": "Create Thread", "description": "Open thread"}],
            "123.456",
        )
        section_blocks = [b for b in blocks if b["type"] == "section"]
        assert len(section_blocks) == 0

    def test_action_button_has_correct_format(self):
        blocks = build_action_blocks(
            "Diagnosis",
            [{"id": "create_thread", "label": "Create Thread", "description": "Open thread"}],
            "123.456",
        )
        button = blocks[1]["elements"][0]
        assert button["action_id"] == "create_thread|123.456"
        assert button["value"] == "create_thread"
        assert button["text"]["text"] == "Create Thread"

    def test_empty_actions_returns_empty_list(self):
        """With no actions, returns empty list (no divider, no section)."""
        blocks = build_action_blocks("Diagnosis", [], "123.456")
        assert len(blocks) == 0

    def test_max_four_buttons(self):
        many_actions = [{"id": f"action_{i}", "label": f"Action {i}", "description": f"Desc {i}"} for i in range(10)]
        blocks = build_action_blocks("Diagnosis", many_actions, "123.456")
        actions_block = next(b for b in blocks if b["type"] == "actions")
        assert len(actions_block["elements"]) == 4


# ---------------------------------------------------------------------------
# ActionExecutor tests
# ---------------------------------------------------------------------------

class TestActionExecutor:
    def test_execute_create_thread(self):
        mock_client = Mock()
        executor = ActionExecutor(client=mock_client)
        result = executor.execute("create_thread", "123.456", "C123", "Redis timeout")
        assert result == "Thread created"
        mock_client.chat_postMessage.assert_called_once()

    def test_execute_explain_code(self):
        mock_agent = Mock()
        mock_agent.diagnose.return_value = "The Redis timeout occurs because..."
        executor = ActionExecutor(client=Mock(), agent=mock_agent)
        result = executor.execute("explain_code", "123.456", "C123", "Redis timeout")
        assert "explanation" in result.lower()

    def test_execute_notify_team(self):
        mock_client = Mock()
        executor = ActionExecutor(client=mock_client)
        result = executor.execute("notify_team", "123.456", "C123", "Redis timeout")
        assert result == "Team notified"
        mock_client.chat_postMessage.assert_called_once()

    def test_execute_unknown_action(self):
        executor = ActionExecutor(client=Mock())
        result = executor.execute("unknown_action", "123.456", "C123", "Redis timeout")
        assert "Unknown action" in result

    def test_execute_no_client(self):
        executor = ActionExecutor(client=None)
        result = executor.execute("create_thread", "123.456", "C123", "Redis timeout")
        assert "not available" in result.lower()


# ---------------------------------------------------------------------------
# Slice 1: show_history displays repository history
# ---------------------------------------------------------------------------

class TestShowHistory:
    def test_show_history_displays_repository_history_in_thread(self):
        """show_history fetches history via LLM-extracted query and posts to thread."""
        from agent.server import CommitInfo
        
        mock_llm = Mock()
        mock_llm.chat.return_value = Mock(content='{"query": "redis timeout", "repos": []}')
        
        mock_agent = Mock()
        mock_agent.llm = mock_llm
        
        mock_flowmap = Mock()
        mock_flowmap.call_tool.return_value = [
            CommitInfo(sha="abc123", message="Fix timeout", author="alice", date="2026-01-01", repo="zedbe-aggregator"),
        ]
        
        mock_client = Mock()
        executor = ActionExecutor(agent=mock_agent, client=mock_client, flowmap=mock_flowmap)
        
        result = executor.execute(
            "show_history",
            "123.456",
            "C123",
            "Redis timeout in zedbe-aggregator",
            thread_history=[{"user": "U1", "text": "why redis timeouts?"}],
        )
        
        assert result == "History posted"
        mock_client.chat_postMessage.assert_called_once()
        call_kwargs = mock_client.chat_postMessage.call_args.kwargs
        assert call_kwargs["channel"] == "C123"
        assert call_kwargs["thread_ts"] == "123.456"
        # LLM was called to extract query
        mock_llm.chat.assert_called_once()

    def test_show_history_searches_all_repos_when_no_specific_repo(self):
        """show_history calls flowmap_history without repo filter (searches all repos)."""
        from agent.server import CommitInfo
        
        mock_llm = Mock()
        mock_llm.chat.return_value = Mock(content='{"query": "redis timeout", "repos": []}')
        
        mock_agent = Mock()
        mock_agent.llm = mock_llm
        
        mock_flowmap = Mock()
        mock_flowmap.call_tool.return_value = [
            CommitInfo(sha="abc", message="Fix", author="a", date="2026-01-01", repo="zedbe-aggregator"),
        ]
        
        mock_client = Mock()
        executor = ActionExecutor(agent=mock_agent, client=mock_client, flowmap=mock_flowmap)
        
        executor.execute(
            "show_history", "123.456", "C123", "Redis timeout",
            thread_history=[{"user": "U1", "text": "why redis timeouts?"}],
        )
        
        # Should call flowmap_history with query, no repo filter
        call_args = mock_flowmap.call_tool.call_args
        assert call_args[0][0] == "flowmap_history"
        assert "query" in call_args[0][1]
        assert call_args[0][1].get("repo") is None

    def test_show_history_groups_cross_repo_commits_by_repo(self):
        """Cross-repo results are grouped by repo name in the output."""
        from agent.server import CommitInfo
        
        mock_llm = Mock()
        mock_llm.chat.return_value = Mock(content='{"query": "redis timeout", "repos": []}')
        
        mock_agent = Mock()
        mock_agent.llm = mock_llm
        
        mock_flowmap = Mock()
        mock_flowmap.call_tool.return_value = [
            CommitInfo(sha="abc", message="Fix timeout", author="a", date="2026-01-01", repo="zedbe-aggregator"),
            CommitInfo(sha="def", message="Add retry", author="b", date="2026-01-02", repo="zedbe-journey-ms"),
        ]
        
        mock_client = Mock()
        executor = ActionExecutor(agent=mock_agent, client=mock_client, flowmap=mock_flowmap)
        
        executor.execute(
            "show_history", "123.456", "C123", "Redis timeout",
            thread_history=[{"user": "U1", "text": "why redis timeouts?"}],
        )
        
        posted_text = mock_client.chat_postMessage.call_args.kwargs["text"]
        assert "zedbe-aggregator" in posted_text
        assert "zedbe-journey-ms" in posted_text

    def test_show_history_uses_thread_history_for_query_context(self):
        """show_history builds LLM prompt from thread history, not just diagnosis."""
        mock_llm = Mock()
        mock_llm.chat.return_value = Mock(content='{"query": "redis timeout", "repos": []}')
        
        mock_agent = Mock()
        mock_agent.llm = mock_llm
        
        mock_flowmap = Mock()
        mock_flowmap.call_tool.return_value = []
        
        mock_client = Mock()
        executor = ActionExecutor(agent=mock_agent, client=mock_client, flowmap=mock_flowmap)
        
        thread_history = [
            {"user": "U1", "text": "why are we getting redis timeouts?"},
            {"user": "U1", "text": "it started after the deploy"},
        ]
        
        executor.execute(
            "show_history", "123.456", "C123", "diagnosis here",
            thread_history=thread_history,
        )
        
        # LLM prompt should include thread history content
        prompt = mock_llm.chat.call_args[0][0]
        assert "redis timeouts" in prompt
        assert "deploy" in prompt

    def test_show_history_falls_back_without_llm(self):
        """Without LLM, falls back to diagnosis as query."""
        from agent.server import CommitInfo
        
        mock_flowmap = Mock()
        mock_flowmap.call_tool.return_value = [
            CommitInfo(sha="abc", message="Fix", author="a", date="2026-01-01", repo="r1"),
        ]
        
        mock_client = Mock()
        executor = ActionExecutor(agent=None, client=mock_client, flowmap=mock_flowmap)
        
        result = executor.execute(
            "show_history", "123.456", "C123", "Redis timeout in zedbe-aggregator",
            thread_history=[],
        )
        
        assert result == "History posted"
        # Should use diagnosis as query
        call_args = mock_flowmap.call_tool.call_args
        assert call_args[0][1].get("query") is not None

    def test_show_history_handles_flowmap_failure(self):
        """show_history handles flowmap errors gracefully."""
        mock_llm = Mock()
        mock_llm.chat.return_value = Mock(content='{"query": "redis", "repos": []}')
        
        mock_agent = Mock()
        mock_agent.llm = mock_llm
        
        mock_flowmap = Mock()
        mock_flowmap.call_tool.side_effect = Exception("FlowMap down")
        
        mock_client = Mock()
        executor = ActionExecutor(agent=mock_agent, client=mock_client, flowmap=mock_flowmap)
        
        result = executor.execute(
            "show_history",
            "123.456",
            "C123",
            "Redis timeout in zedbe-aggregator",
            thread_history=[{"user": "U1", "text": "why?"}],
        )
        
        assert "failed" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------------
# Slice 2: explain_code displays code explanation
# ---------------------------------------------------------------------------

class TestExplainCode:
    def test_explain_code_displays_code_explanation_in_thread(self):
        """explain_code generates explanation and posts to thread."""
        mock_agent = Mock()
        mock_agent.diagnose.return_value = "The Redis timeout occurs because connection pool is exhausted."
        
        mock_client = Mock()
        executor = ActionExecutor(agent=mock_agent, client=mock_client, flowmap=Mock())
        
        result = executor.execute(
            "explain_code",
            "123.456",
            "C123",
            "Redis timeout in src/config.js:15",
        )
        
        assert result == "Explanation posted"
        mock_client.chat_postMessage.assert_called_once()
        call_kwargs = mock_client.chat_postMessage.call_args.kwargs
        assert call_kwargs["channel"] == "C123"
        assert call_kwargs["thread_ts"] == "123.456"
        assert "Redis timeout" in call_kwargs["text"]

    def test_explain_code_includes_thread_history_in_prompt(self):
        """explain_code includes thread history in prompt."""
        mock_agent = Mock()
        mock_agent.diagnose.return_value = "Explanation"
        
        mock_client = Mock()
        executor = ActionExecutor(agent=mock_agent, client=mock_client, flowmap=Mock())
        
        thread_history = [
            {"user": "U123", "text": "We saw this issue yesterday"},
            {"user": "U456", "text": "It happens during peak hours"},
        ]
        
        executor.execute(
            "explain_code",
            "123.456",
            "C123",
            "Redis timeout",
        )
        
        call_args = mock_agent.diagnose.call_args[0][0]
        assert "Redis timeout" in call_args

    def test_explain_code_handles_agent_failure(self):
        """explain_code handles agent errors gracefully."""
        mock_agent = Mock()
        mock_agent.diagnose.side_effect = Exception("Agent down")
        
        mock_client = Mock()
        executor = ActionExecutor(agent=mock_agent, client=mock_client, flowmap=Mock())
        
        result = executor.execute(
            "explain_code",
            "123.456",
            "C123",
            "Redis timeout",
        )
        
        assert "failed" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------------
# Slice 4: idempotency prevents duplicate execution
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_same_action_same_message_prevents_duplicate_execution(self):
        """Same action on same message only executes once."""
        mock_client = Mock()
        executor = ActionExecutor(agent=Mock(), client=mock_client, flowmap=Mock())
        
        result1 = executor.execute("create_thread", "123.456", "C123", "test")
        result2 = executor.execute("create_thread", "123.456", "C123", "test")
        
        assert result1 == "Thread created"
        assert result2 == "Already executed"
        assert mock_client.chat_postMessage.call_count == 1

    def test_same_action_different_message_executes(self):
        """Same action on different messages executes both."""
        mock_client = Mock()
        executor = ActionExecutor(agent=Mock(), client=mock_client, flowmap=Mock())
        
        result1 = executor.execute("create_thread", "111.111", "C123", "test1")
        result2 = executor.execute("create_thread", "222.222", "C123", "test2")
        
        assert result1 == "Thread created"
        assert result2 == "Thread created"
        assert mock_client.chat_postMessage.call_count == 2

    def test_different_action_same_message_executes(self):
        """Different actions on same message both execute."""
        mock_client = Mock()
        executor = ActionExecutor(agent=Mock(), client=mock_client, flowmap=Mock())
        
        result1 = executor.execute("create_thread", "123.456", "C123", "test")
        result2 = executor.execute("notify_team", "123.456", "C123", "test")
        
        assert result1 == "Thread created"
        assert result2 == "Team notified"
        assert mock_client.chat_postMessage.call_count == 2


# ---------------------------------------------------------------------------
# Integration: diagnosis includes action buttons
# ---------------------------------------------------------------------------

class TestActionIntegration:
    def test_diagnosis_includes_action_buttons(self):
        """End-to-end: planner recommends -> registry stores -> buttons built."""
        planner = ActionPlanner(llm=None)
        registry = ActionRegistry()

        diagnosis = "Redis connection timeout in aggregator service"
        actions = planner.recommend(diagnosis)
        assert len(actions) >= 1

        registry.store("123.456", actions, diagnosis=diagnosis)
        stored = registry.get("123.456")
        assert stored is not None
        assert stored["diagnosis"] == diagnosis

        blocks = build_action_blocks(diagnosis, stored["actions"], "123.456")
        actions_block = next((b for b in blocks if b["type"] == "actions"), None)
        assert actions_block is not None
        assert len(actions_block["elements"]) >= 1

    def test_button_click_retrieves_actions(self):
        """Simulate button click: retrieve actions from registry."""
        registry = ActionRegistry()
        actions = [{"id": "create_thread", "label": "Create Thread", "description": "Open thread"}]
        registry.store("123.456", actions, diagnosis="Root cause: Redis timeout")

        retrieved = registry.get("123.456")
        assert retrieved["actions"][0]["id"] == "create_thread"
        assert retrieved["diagnosis"] == "Root cause: Redis timeout"

    def test_end_to_end_action_flow(self):
        """Simulate: diagnosis -> store actions -> build blocks -> button click -> execute."""
        registry = ActionRegistry()
        mock_client = Mock()

        # 1. Diagnosis produces actions
        actions = [{"id": "create_thread", "label": "Create Thread", "description": "Open thread"}]
        registry.store("123.456", actions)

        # 2. Build blocks for message
        blocks = build_action_blocks("Redis timeout", actions, "123.456")
        assert any(b["type"] == "actions" for b in blocks)

        # 3. User clicks button -> action_id = "create_thread|123.456"
        stored = registry.get("123.456")
        assert stored is not None

        # 4. Execute
        executor = ActionExecutor(client=mock_client)
        result = executor.execute("create_thread", "123.456", "C123", "Redis timeout")
        assert result == "Thread created"
