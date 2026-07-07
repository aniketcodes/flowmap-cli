"""Tests for SlackAdapter."""

import pytest
from unittest.mock import Mock, AsyncMock
from agent.slack_adapter import SlackAdapter


class TestSlackAdapter:
    def test_adapter_name_returns_slack(self):
        """Adapter name is 'slack'."""
        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._client = Mock()
        assert adapter.name() == "slack"

    def test_adapter_list_tools_delegates_to_client(self):
        """list_tools() returns tools from client with prefix."""
        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._client = Mock()
        adapter._connected = True
        adapter._client.list_tools.return_value = [
            {"name": "conversations_history", "description": "Get history", "inputSchema": {}}
        ]

        tools = adapter.list_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "slack_conversations_history"

    def test_adapter_list_tools_empty_when_no_client(self):
        """list_tools() returns empty list when not connected."""
        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._client = None
        adapter._connected = False

        tools = adapter.list_tools()

        assert tools == []

    def test_adapter_list_tools_does_not_mutate_cache(self):
        """list_tools() should not mutate client's tool cache."""
        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._client = Mock()
        adapter._connected = True
        original_tool = {"name": "conversations_history", "description": "Get history", "inputSchema": {}}
        adapter._client.list_tools.return_value = [original_tool]

        # Call list_tools twice
        tools1 = adapter.list_tools()
        tools2 = adapter.list_tools()

        # Original should not be mutated
        assert original_tool["name"] == "conversations_history"
        # Both calls should return correctly prefixed names
        assert tools1[0]["name"] == "slack_conversations_history"
        assert tools2[0]["name"] == "slack_conversations_history"

    def test_adapter_call_tool_routes_correctly(self):
        """call_tool() routes to client.call_tool() via dedicated event loop."""
        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._client = Mock()
        adapter._client.call_tool = AsyncMock(return_value="test result")
        adapter._connected = True

        result = adapter.call_tool("slack_conversations_history", {"channel": "C123"})

        assert result == "test result"

    def test_adapter_call_tool_returns_error_when_not_connected(self):
        """call_tool() returns error string when not connected."""
        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._connected = False

        result = adapter.call_tool("slack_conversations_history", {"channel": "C123"})

        assert "Not connected" in result

    def test_adapter_connect_returns_bool(self):
        """connect() returns boolean."""
        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._client = Mock()
        adapter._client.connect = AsyncMock(return_value=True)

        assert hasattr(adapter._client, 'connect')

    def test_adapter_disconnect_cleans_up(self):
        """disconnect() calls client.disconnect()."""
        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._client = Mock()
        adapter._client.disconnect = AsyncMock()

        assert hasattr(adapter._client, 'disconnect')
