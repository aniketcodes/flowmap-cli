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
    def test_builds_section_and_actions(self):
        blocks = build_action_blocks(
            "Root cause: Redis timeout",
            [{"id": "create_thread", "label": "Create Thread", "description": "Open thread"}],
            "123.456",
        )
        assert len(blocks) >= 2
        assert blocks[0]["type"] == "section"
        assert blocks[1]["type"] == "divider"
        assert blocks[2]["type"] == "actions"

    def test_action_button_has_correct_format(self):
        blocks = build_action_blocks(
            "Diagnosis",
            [{"id": "create_thread", "label": "Create Thread", "description": "Open thread"}],
            "123.456",
        )
        button = blocks[2]["elements"][0]
        assert button["action_id"] == "create_thread|123.456"
        assert button["value"] == "create_thread"
        assert button["text"]["text"] == "Create Thread"

    def test_empty_actions_no_actions_block(self):
        blocks = build_action_blocks("Diagnosis", [], "123.456")
        assert len(blocks) == 2
        assert all(b["type"] != "actions" for b in blocks)

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
        executor = ActionExecutor(client=Mock())
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
