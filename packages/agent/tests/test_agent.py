"""Tests for Agent Logic."""

import pytest
from unittest.mock import Mock, patch

from agent.server import FlowMapMCPServer


# Shared tool schema for mocks
_FLOWMAP_TOOLS = FlowMapMCPServer.list_tools(FlowMapMCPServer.__new__(FlowMapMCPServer))


class TestAgent:
    """Test agent tool calling and agentic loop."""

    def test_agent_can_be_created(self):
        """Agent can be instantiated."""
        from agent.agent import Agent
        agent = Agent()
        assert agent is not None

    def test_agent_diagnose_returns_response(self):
        """Diagnose returns a response."""
        from agent.agent import Agent
        mock_llm = Mock()
        # LLM must read a file before FINAL_ANSWER (cat_calls guard)
        mock_llm.chat.side_effect = [
            Mock(content='TOOL_CALL: flowmap_cat(repo="test", file="test.ts")'),
            Mock(content="FINAL_ANSWER: test answer"),
        ]
        mock_mcp = Mock()
        mock_mcp.list_tools.return_value = _FLOWMAP_TOOLS
        mock_mcp.search.return_value = []
        mock_mcp.history.return_value = []
        mock_mcp.cat.return_value = "1: test code"
        agent = Agent(mcp=mock_mcp, llm=mock_llm)
        response = agent.diagnose("hello")
        assert response == "test answer"

    def test_agent_sends_prompt_to_llm(self):
        """Agent sends tool definitions + query to LLM."""
        from agent.agent import Agent
        mock_llm = Mock()
        mock_llm.chat.return_value = Mock(content="FINAL_ANSWER: done")
        mock_mcp = Mock()
        mock_mcp.list_tools.return_value = _FLOWMAP_TOOLS
        mock_mcp.search.return_value = []
        agent = Agent(mcp=mock_mcp, llm=mock_llm)

        agent.diagnose("test query")

        call_args = mock_llm.chat.call_args[0][0]
        assert "flowmap_search" in call_args
        assert "test query" in call_args

    def test_agent_executes_tool_call(self):
        """Agent executes TOOL_CALL from LLM and feeds result back."""
        from agent.agent import Agent

        mock_mcp = Mock()
        mock_mcp.list_tools.return_value = _FLOWMAP_TOOLS
        mock_mcp.search.return_value = [
            Mock(repo="r", file="f.ts", start_line=1, end_line=10,
                 text="code here", score=0.9, signature="fn()")
        ]
        mock_mcp.cat.return_value = "1: code here"
        mock_llm = Mock()
        # First call: LLM asks to search. Second: LLM reads file. Third: LLM gives final answer.
        mock_llm.chat.side_effect = [
            Mock(content='TOOL_CALL: flowmap_search(query="test")'),
            Mock(content='TOOL_CALL: flowmap_cat(repo="r", file="f.ts")'),
            Mock(content="FINAL_ANSWER: found it in f.ts:1"),
        ]

        agent = Agent(mcp=mock_mcp, llm=mock_llm)
        response = agent.diagnose("test")

        assert response == "found it in f.ts:1"
        assert mock_mcp.search.call_count == 1
        assert mock_mcp.cat.call_count == 1
        assert mock_llm.chat.call_count == 3

    def test_agent_multi_step_loop(self):
        """Agent can make multiple tool calls before answering."""
        from agent.agent import Agent

        mock_mcp = Mock()
        mock_mcp.list_tools.return_value = _FLOWMAP_TOOLS
        mock_mcp.search.return_value = [Mock(
            repo="test-repo", file="auth.ts", start_line=10, end_line=20,
            text="function authenticate() {}", score=0.9, signature="authenticate()",
        )]
        mock_mcp.cat.return_value = "10: function authenticate() { return true; }"
        mock_llm = Mock()
        mock_llm.chat.side_effect = [
            Mock(content='TOOL_CALL: flowmap_search(query="auth")'),
            Mock(content='TOOL_CALL: flowmap_cat(repo="test-repo", file="auth.ts")'),
            Mock(content="FINAL_ANSWER: Authentication is in auth.ts:10"),
        ]

        agent = Agent(mcp=mock_mcp, llm=mock_llm)
        response = agent.diagnose("how does auth work")

        assert "auth.ts" in response
        assert mock_mcp.search.call_count == 1
        assert mock_mcp.cat.call_count == 1
        assert mock_llm.chat.call_count == 3

    def test_agent_handles_tool_error(self):
        """Agent handles tool errors gracefully."""
        from agent.agent import Agent

        mock_mcp = Mock()
        mock_mcp.list_tools.return_value = _FLOWMAP_TOOLS
        mock_mcp.search.side_effect = Exception("search failed")
        mock_mcp.cat.return_value = "1: some code"
        mock_llm = Mock()
        mock_llm.chat.side_effect = [
            Mock(content='TOOL_CALL: flowmap_search(query="test")'),
            Mock(content='TOOL_CALL: flowmap_cat(repo="test", file="test.ts")'),
            Mock(content="FINAL_ANSWER: encountered an error but here's what I know"),
        ]

        agent = Agent(mcp=mock_mcp, llm=mock_llm)
        response = agent.diagnose("test")

        # Should get error in tool result, then final answer
        assert response is not None
        assert len(response) > 0

    def test_agent_no_llm_returns_fallback(self):
        """Without LLM, returns fallback."""
        from agent.agent import Agent
        agent = Agent()
        response = agent.diagnose("test")
        assert "No LLM" in response

    def test_agent_max_steps_limit(self):
        """Agent stops after max steps."""
        from agent.agent import Agent

        mock_llm = Mock()
        # Always return TOOL_CALL to test the limit
        mock_llm.chat.return_value = Mock(content='TOOL_CALL: flowmap_search(query="loop")')
        mock_mcp = Mock()
        mock_mcp.list_tools.return_value = _FLOWMAP_TOOLS
        mock_mcp.search.return_value = []

        agent = Agent(mcp=mock_mcp, llm=mock_llm)
        response = agent.diagnose("loop")

        assert "max steps" in response.lower()
        # Should be 15 steps (the max)
        assert mock_llm.chat.call_count == 15
