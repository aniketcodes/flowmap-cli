"""Tests for Component Wiring."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from unittest.mock import Mock, patch
from agent.bot import SlackBot
from agent.agent import Agent
from agent.server import FlowMapMCPServer
from agent.llm import LLMClient


# Shared tool schema for mocks
_FLOWMAP_TOOLS = FlowMapMCPServer.list_tools(FlowMapMCPServer.__new__(FlowMapMCPServer))


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
    """Test that Agent calls MCP Server."""

    def test_agent_wires_to_mcp(self):
        """Agent calls MCP tools when LLM requests them."""
        mock_mcp = Mock(spec=FlowMapMCPServer)
        mock_mcp.list_tools.return_value = _FLOWMAP_TOOLS
        mock_mcp.search.return_value = []
        mock_mcp.history.return_value = []
        mock_mcp.cat.return_value = "1: test code"
        mock_llm = Mock(spec=LLMClient)
        mock_llm.chat.side_effect = [
            Mock(content='TOOL_CALL: flowmap_search(query="test")'),
            Mock(content='TOOL_CALL: flowmap_cat(repo="test", file="test.ts")'),
            Mock(content="FINAL_ANSWER: test"),
        ]

        agent = Agent(mcp=mock_mcp, llm=mock_llm)
        agent.diagnose("402 errors")

        mock_mcp.search.assert_called_once()
        mock_mcp.cat.assert_called_once()


class TestAgentLLMWiring:
    """Test that Agent calls LLM."""

    def test_agent_wires_to_llm(self):
        """Agent calls LLM.chat with agentic prompt."""
        mock_llm = Mock(spec=LLMClient)
        # LLM must read a file before FINAL_ANSWER (cat_calls guard)
        mock_llm.chat.side_effect = [
            Mock(content='TOOL_CALL: flowmap_cat(repo="test", file="test.ts")'),
            Mock(content="FINAL_ANSWER: explanation"),
        ]
        mock_mcp = Mock(spec=FlowMapMCPServer)
        mock_mcp.list_tools.return_value = _FLOWMAP_TOOLS
        mock_mcp.search.return_value = []
        mock_mcp.cat.return_value = "1: test code"

        agent = Agent(mcp=mock_mcp, llm=mock_llm)
        result = agent.diagnose("402 errors")

        assert "explanation" in result
        mock_llm.chat.assert_called()


class TestFullFlow:
    """Test end-to-end flow: message → agent → tools → LLM → response."""

    def test_full_flow_returns_explanation(self):
        """Complete flow from message to response."""
        bot = SlackBot()
        agent = Agent()
        mcp = FlowMapMCPServer()
        llm = LLMClient(provider="mock")

        bot.agent = agent
        agent.mcp = mcp
        agent.llm = llm

        # Register the mcp as an adapter for proper routing
        from agent.flowmap_adapter import FlowMapAdapter
        agent.adapters = [FlowMapAdapter(server=mcp)]
        agent._tool_routes = {}
        for adapter in agent.adapters:
            for tool in adapter.list_tools():
                agent._tool_routes[tool["name"]] = adapter

        with patch.object(mcp, 'search') as mock_search:
            mock_search.return_value = [
                Mock(file="app.py", score=0.95, text="retry logic removed",
                     repo="test", start_line=1, end_line=10, signature="retry()")
            ]

            response = bot.handle_mention("Why 402 errors?")

        assert response is not None
        assert len(response) > 0
