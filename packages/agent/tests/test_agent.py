"""Tests for Agent Logic."""

import pytest
from unittest.mock import Mock

from agent.agent import Agent
from agent.llm import LLMResponse, ToolCall
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


_FLOWMAP_TOOLS = [
    {"name": "flowmap_search", "description": "Search code", "inputSchema": {
        "properties": {"query": {"type": "string"}, "mode": {"type": "string"}},
        "required": ["query"]
    }},
    {"name": "flowmap_cat", "description": "Read a file", "inputSchema": {
        "properties": {"repo": {"type": "string"}, "file": {"type": "string"}},
        "required": ["repo", "file"]
    }},
    {"name": "flowmap_history", "description": "Git history", "inputSchema": {
        "properties": {"repo": {"type": "string"}, "file": {"type": "string"}},
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


class TestAgent:
    """Test agent tool calling and agentic loop."""

    def test_agent_can_be_created(self):
        """Agent can be instantiated."""
        agent = Agent()
        assert agent is not None

    def test_agent_diagnose_returns_response(self):
        """Diagnose returns a response after sufficient tool calls."""
        search_handler = Mock(return_value="[]")
        cat_handler = Mock(return_value="1: test code")

        adapter = MockAdapter("flowmap", _FLOWMAP_TOOLS, handler=lambda n, a: (
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
            LLMResponse(content="test answer", tool_calls=None),
        ]

        agent = Agent(adapters=[adapter], llm=mock_llm)
        response = agent.diagnose("hello")
        assert response == "test answer"

    def test_agent_sends_tools_to_llm(self):
        """Agent sends OpenAI-format tools + query to LLM."""
        adapter = MockAdapter("flowmap", _FLOWMAP_TOOLS)
        mock_llm = Mock()
        mock_llm.chat.return_value = LLMResponse(content="done", tool_calls=None)
        agent = Agent(adapters=[adapter], llm=mock_llm)

        agent.diagnose("test query")

        # First call should have tools and user query
        first_call_kwargs = mock_llm.chat.call_args_list[0][1]
        assert "tools" in first_call_kwargs
        assert "messages" in first_call_kwargs
        user_messages = [m for m in first_call_kwargs["messages"] if m["role"] == "user"]
        assert user_messages[0]["content"] == "test query"

    def test_agent_executes_tool_calls(self):
        """Agent executes native tool calls and feeds results back."""
        search_handler = Mock(return_value='[{"repo":"r","file":"f.ts","start_line":1,"end_line":10,"text":"code","score":0.9}]')
        cat_handler = Mock(return_value="1: code here")

        adapter = MockAdapter("flowmap", _FLOWMAP_TOOLS, handler=lambda n, a: (
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
            LLMResponse(content="found it in f.ts:1", tool_calls=None),
        ]

        agent = Agent(adapters=[adapter], llm=mock_llm)
        response = agent.diagnose("test")

        assert response == "found it in f.ts:1"
        assert search_handler.call_count == 1
        assert cat_handler.call_count == 1
        assert mock_llm.chat.call_count == 3

    def test_agent_multi_step_loop(self):
        """Agent can make multiple tool calls before answering."""
        search_handler = Mock(return_value='[{"repo":"r","file":"auth.ts","start_line":10,"end_line":20,"text":"fn auth","score":0.9}]')
        cat_handler = Mock(return_value="10: function authenticate() { return true; }")

        adapter = MockAdapter("flowmap", _FLOWMAP_TOOLS, handler=lambda n, a: (
            search_handler(n, a) if n == "flowmap_search" else cat_handler(n, a)
        ))

        mock_llm = Mock()
        mock_llm.chat.side_effect = [
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc_1", name="flowmap_search", arguments={"query": "auth"})
            ]),
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc_2", name="flowmap_cat", arguments={"repo": "r", "file": "auth.ts"})
            ]),
            LLMResponse(content="Authentication is in auth.ts:10", tool_calls=None),
        ]

        agent = Agent(adapters=[adapter], llm=mock_llm)
        response = agent.diagnose("how does auth work")

        assert "auth.ts" in response
        assert search_handler.call_count == 1
        assert cat_handler.call_count == 1
        assert mock_llm.chat.call_count == 3

    def test_agent_handles_tool_error(self):
        """Agent handles tool errors gracefully."""
        search_handler = Mock(side_effect=Exception("search failed"))
        cat_handler = Mock(return_value="1: some code")

        adapter = MockAdapter("flowmap", _FLOWMAP_TOOLS, handler=lambda n, a: (
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
            LLMResponse(content="encountered an error but here's what I know", tool_calls=None),
        ]

        agent = Agent(adapters=[adapter], llm=mock_llm)
        response = agent.diagnose("test")

        assert response is not None
        assert len(response) > 0

    def test_agent_no_llm_returns_fallback(self):
        """Without LLM, returns fallback."""
        agent = Agent()
        response = agent.diagnose("test")
        assert "No LLM" in response

    def test_agent_max_steps_limit(self):
        """Agent stops after max steps."""
        adapter = MockAdapter("flowmap", _FLOWMAP_TOOLS, handler=lambda n, a: "result")

        mock_llm = Mock()
        # Always return tool calls to test the limit
        mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[
            ToolCall(id="tc_loop", name="flowmap_search", arguments={"query": "loop"})
        ])

        agent = Agent(adapters=[adapter], llm=mock_llm)
        response = agent.diagnose("loop", max_steps=5)

        assert "max steps" in response.lower()
        assert mock_llm.chat.call_count == 5

    def test_agent_strips_final_answer_prefix(self):
        """Agent strips FINAL_ANSWER: prefix from LLM response."""
        adapter = MockAdapter("flowmap", _FLOWMAP_TOOLS, handler=lambda n, a: "result")

        mock_llm = Mock()
        mock_llm.chat.side_effect = [
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc_1", name="flowmap_search", arguments={"query": "x"})
            ]),
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc_2", name="flowmap_search", arguments={"query": "y"})
            ]),
            LLMResponse(content="FINAL_ANSWER: the answer is 42", tool_calls=None),
        ]

        agent = Agent(adapters=[adapter], llm=mock_llm)
        response = agent.diagnose("test")

        assert response == "the answer is 42"
        assert "FINAL_ANSWER" not in response
