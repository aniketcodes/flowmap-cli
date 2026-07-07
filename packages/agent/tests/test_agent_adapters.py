"""Tests for Agent adapter integration."""

import pytest
from unittest.mock import Mock
from agent.agent import Agent
from agent.adapter import MCPAdapter


class MockAdapter(MCPAdapter):
    """Test adapter with mock tools."""

    def __init__(self, adapter_name: str, tools: list[dict], handler=None):
        self._name = adapter_name
        self._tools = tools
        self._handler = handler or (lambda name, args: f"Result from {name}")

    def name(self) -> str:
        return self._name

    def list_tools(self) -> list[dict]:
        return self._tools

    def call_tool(self, name: str, args: dict) -> str:
        return self._handler(name, args)


class TestAgentAdapters:
    def test_agent_accepts_adapters_list(self):
        """Agent accepts list of adapters."""
        adapter = MockAdapter("test", [{"name": "test_tool", "description": "Test"}])
        agent = Agent(adapters=[adapter], llm=Mock())

        assert len(agent.adapters) == 1

    def test_agent_builds_routing_table_on_init(self):
        """Agent builds tool routing table once in __init__."""
        adapter1 = MockAdapter("flowmap", [
            {"name": "flowmap_search", "description": "Search code"}
        ])
        adapter2 = MockAdapter("slack", [
            {"name": "slack_history", "description": "Get history"}
        ])

        agent = Agent(adapters=[adapter1, adapter2], llm=Mock())

        assert "flowmap_search" in agent._tool_routes
        assert "slack_history" in agent._tool_routes
        assert agent._tool_routes["flowmap_search"] is adapter1
        assert agent._tool_routes["slack_history"] is adapter2

    def test_agent_build_tool_prompt_dynamically(self):
        """Agent generates tool prompt from adapters, not hardcoded string."""
        adapter = MockAdapter("flowmap", [
            {"name": "flowmap_search", "description": "Search code", "inputSchema": {
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"]
            }}
        ])

        agent = Agent(adapters=[adapter], llm=Mock())
        prompt = agent._build_tool_prompt()

        assert "flowmap_search" in prompt
        assert "Search code" in prompt
        assert "query" in prompt

    def test_agent_execute_tool_routes_to_correct_adapter(self):
        """Agent routes tool call to correct adapter via routing table."""
        flowmap_handler = Mock(return_value="FlowMap result")
        slack_handler = Mock(return_value="Slack result")

        adapter1 = MockAdapter("flowmap",
            [{"name": "flowmap_search", "description": "Search"}],
            handler=flowmap_handler
        )
        adapter2 = MockAdapter("slack",
            [{"name": "slack_history", "description": "History"}],
            handler=slack_handler
        )

        agent = Agent(adapters=[adapter1, adapter2], llm=Mock())

        # Test routing to flowmap
        result = agent._execute_tool("flowmap_search(query='test')")
        flowmap_handler.assert_called_once()
        assert result == "FlowMap result"

        # Reset and test routing to slack
        flowmap_handler.reset_mock()
        result = agent._execute_tool("slack_history(channel='C123')")
        slack_handler.assert_called_once()
        assert result == "Slack result"

    def test_agent_unknown_tool_returns_error(self):
        """Agent returns error for unknown tool."""
        adapter = MockAdapter("test", [{"name": "test_tool", "description": "Test"}])
        agent = Agent(adapters=[adapter], llm=Mock())

        result = agent._execute_tool("unknown_tool()")
        assert "Unknown tool" in result

    def test_agent_backward_compatible_with_mcp_param(self):
        """Agent still works with old mcp= parameter for migration."""
        mock_mcp = Mock()
        mock_mcp.list_tools.return_value = [
            {"name": "flowmap_search", "description": "Search"}
        ]

        agent = Agent(mcp=mock_mcp, llm=Mock())
        prompt = agent._build_tool_prompt()

        assert "flowmap_search" in prompt
        assert "Search" in prompt
