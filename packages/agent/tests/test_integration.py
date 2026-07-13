"""Tests for Component Wiring."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from unittest.mock import Mock, patch
from agent.bot import SlackBot
from agent.agent import Agent
from agent.llm import LLMClient, LLMResponse, ToolCall
from agent.adapter import MCPAdapter


class MockAdapter(MCPAdapter):
    def __init__(self, tools, handler=None):
        self._tools = tools
        self._handler = handler or (lambda n, a: "result")

    def name(self):
        return "flowmap"

    def list_tools(self):
        return self._tools

    def call_tool(self, name, args):
        return self._handler(name, args)


_FLOWMAP_TOOLS = [
    {"name": "flowmap_search", "description": "Search code", "inputSchema": {
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    }},
    {"name": "flowmap_cat", "description": "Read file", "inputSchema": {
        "properties": {"repo": {"type": "string"}, "file": {"type": "string"}},
        "required": ["repo", "file"]
    }},
    {"name": "flowmap_history", "description": "Git history", "inputSchema": {
        "properties": {"repo": {"type": "string"}},
        "required": ["repo"]
    }},
    {"name": "flowmap_repos", "description": "List repos", "inputSchema": {
        "properties": {}, "required": []
    }},
    {"name": "flowmap_symbols", "description": "Find symbols", "inputSchema": {
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    }},
    {"name": "flowmap_map", "description": "Map a repo", "inputSchema": {
        "properties": {"repo": {"type": "string"}},
        "required": ["repo"]
    }},
]


class TestBotAgentWiring:
    """Test that Bot calls Agent."""

    def test_bot_wires_to_agent(self):
        """Bot calls Agent.diagnose when handling mention."""
        bot = SlackBot()
        mock_agent = Mock(spec=Agent)
        mock_agent.diagnose.return_value = "test response"
        bot.agent = mock_agent

        bot.handle_mention("Why 402 errors?")

        mock_agent.diagnose.assert_called_once_with("Why 402 errors?")


class TestAgentMCPWiring:
    """Test that Agent calls MCP tools via adapter."""

    def test_agent_wires_to_mcp(self):
        """Agent calls adapter tools when LLM requests them."""
        search_handler = Mock(return_value='[]')
        cat_handler = Mock(return_value="1: test code")

        adapter = MockAdapter(_FLOWMAP_TOOLS, handler=lambda n, a: (
            search_handler(n, a) if n == "flowmap_search" else cat_handler(n, a)
        ))

        mock_llm = Mock()
        mock_llm.chat.side_effect = [
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc_1", name="flowmap_search", arguments={"query": "test"})
            ]),
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc_2", name="flowmap_cat", arguments={"repo": "r", "file": "f.ts"})
            ]),
            LLMResponse(content="test", tool_calls=None),
        ]

        agent = Agent(adapters=[adapter], llm=mock_llm)
        agent.diagnose("402 errors")

        assert search_handler.call_count == 1
        assert cat_handler.call_count == 1


class TestAgentLLMWiring:
    """Test that Agent calls LLM with tools."""

    def test_agent_wires_to_llm(self):
        """Agent calls LLM.chat with tools parameter."""
        adapter = MockAdapter(_FLOWMAP_TOOLS, handler=lambda n, a: "result")

        mock_llm = Mock()
        mock_llm.chat.side_effect = [
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc_1", name="flowmap_search", arguments={"query": "402"})
            ]),
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc_2", name="flowmap_cat", arguments={"repo": "r", "file": "err.ts"})
            ]),
            LLMResponse(content="explanation", tool_calls=None),
        ]

        agent = Agent(adapters=[adapter], llm=mock_llm)
        result = agent.diagnose("402 errors")

        assert "explanation" in result
        assert mock_llm.chat.call_count == 3
        # Verify tools were passed
        for call in mock_llm.chat.call_args_list:
            assert "tools" in call.kwargs


class TestFullFlow:
    """Test end-to-end flow: message → agent → tools → LLM → response."""

    def test_full_flow_returns_explanation(self):
        """Complete flow from message to response."""
        bot = SlackBot()

        adapter = MockAdapter(_FLOWMAP_TOOLS, handler=lambda n, a: "result")

        mock_llm = Mock()
        mock_llm.chat.side_effect = [
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc_1", name="flowmap_search", arguments={"query": "402"})
            ]),
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc_2", name="flowmap_cat", arguments={"repo": "r", "file": "pay.ts"})
            ]),
            LLMResponse(content="402 errors are caused by rate limiting in pay.ts:42", tool_calls=None),
        ]

        agent = Agent(adapters=[adapter], llm=mock_llm)
        bot.agent = agent

        response = bot.handle_mention("Why 402 errors?")

        assert response is not None
        assert len(response) > 0
