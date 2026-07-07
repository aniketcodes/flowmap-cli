"""Tests for MCPAdapter ABC."""

import pytest
from agent.adapter import MCPAdapter


class TestMCPAdapter:
    def test_adapter_cannot_be_instantiated(self):
        """MCPAdapter is abstract — cannot instantiate directly."""
        with pytest.raises(TypeError):
            MCPAdapter()

    def test_incomplete_subclass_raises_typeerror(self):
        """Subclass missing abstract methods raises TypeError on instantiation."""
        class IncompleteAdapter(MCPAdapter):
            def name(self): return "incomplete"
            # Missing list_tools() and call_tool()

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_complete_subclass_can_be_instantiated(self):
        """Subclass implementing all abstract methods can be instantiated."""
        class CompleteAdapter(MCPAdapter):
            def name(self): return "complete"
            def list_tools(self): return []
            def call_tool(self, name, args): return ""

        adapter = CompleteAdapter()
        assert adapter.name() == "complete"

    def test_adapter_connect_returns_true_by_default(self):
        """Default connect() returns True."""
        class MinimalAdapter(MCPAdapter):
            def name(self): return "minimal"
            def list_tools(self): return []
            def call_tool(self, name, args): return ""

        adapter = MinimalAdapter()
        assert adapter.connect() is True

    def test_adapter_disconnect_is_noop(self):
        """Default disconnect() does not raise."""
        class MinimalAdapter(MCPAdapter):
            def name(self): return "minimal"
            def list_tools(self): return []
            def call_tool(self, name, args): return ""

        adapter = MinimalAdapter()
        adapter.disconnect()  # Should not raise

    def test_adapter_list_tools_returns_list(self):
        """list_tools() must return a list."""
        class TestAdapter(MCPAdapter):
            def name(self): return "test"
            def list_tools(self): return [{"name": "tool1"}]
            def call_tool(self, name, args): return ""

        adapter = TestAdapter()
        tools = adapter.list_tools()
        assert isinstance(tools, list)
        assert len(tools) == 1

    def test_adapter_call_tool_returns_string(self):
        """call_tool() must return a string."""
        class TestAdapter(MCPAdapter):
            def name(self): return "test"
            def list_tools(self): return []
            def call_tool(self, name, args): return "result"

        adapter = TestAdapter()
        result = adapter.call_tool("tool", {})
        assert isinstance(result, str)
