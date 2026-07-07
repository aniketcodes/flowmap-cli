"""Tests for FlowMapAdapter."""

import pytest
from agent.server import FlowMapMCPServer
from agent.flowmap_adapter import FlowMapAdapter


class TestFlowMapMCPServerSchema:
    """Verify FlowMapMCPServer returns proper MCP tool schema."""

    def test_list_tools_returns_list_of_dicts(self):
        """list_tools() must return list[dict], not list[str]."""
        server = FlowMapMCPServer()
        tools = server.list_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0
        assert isinstance(tools[0], dict)

    def test_tool_has_required_mcp_fields(self):
        """Each tool must have name, description, inputSchema."""
        server = FlowMapMCPServer()
        tools = server.list_tools()

        for tool in tools:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool missing 'description': {tool}"
            assert "inputSchema" in tool, f"Tool missing 'inputSchema': {tool}"

    def test_search_tool_has_query_parameter(self):
        """flowmap_search must have 'query' as required parameter."""
        server = FlowMapMCPServer()
        tools = server.list_tools()
        search_tool = next(t for t in tools if t["name"] == "flowmap_search")

        assert "query" in search_tool["inputSchema"]["properties"]
        assert "query" in search_tool["inputSchema"]["required"]

    def test_cat_tool_has_repo_file_parameters(self):
        """flowmap_cat must have 'repo' and 'file' as required parameters."""
        server = FlowMapMCPServer()
        tools = server.list_tools()
        cat_tool = next(t for t in tools if t["name"] == "flowmap_cat")

        assert "repo" in cat_tool["inputSchema"]["properties"]
        assert "file" in cat_tool["inputSchema"]["properties"]
        assert "repo" in cat_tool["inputSchema"]["required"]
        assert "file" in cat_tool["inputSchema"]["required"]

    def test_all_six_tools_present(self):
        """All 6 FlowMap tools must be present."""
        server = FlowMapMCPServer()
        tools = server.list_tools()
        tool_names = {t["name"] for t in tools}

        expected = {"flowmap_search", "flowmap_cat", "flowmap_history",
                    "flowmap_repos", "flowmap_symbols", "flowmap_map"}
        assert tool_names == expected


class TestFlowMapAdapter:
    """Test FlowMapAdapter wrapping FlowMapMCPServer."""

    def test_adapter_name_returns_flowmap(self):
        """Adapter name is 'flowmap'."""
        adapter = FlowMapAdapter()
        assert adapter.name() == "flowmap"

    def test_adapter_list_tools_delegates_to_server(self):
        """list_tools() returns same as server."""
        adapter = FlowMapAdapter()
        tools = adapter.list_tools()

        assert len(tools) == 6
        assert all(isinstance(t, dict) for t in tools)

    def test_adapter_call_tool_routes_search(self):
        """call_tool('flowmap_search', {...}) routes to server.search()."""
        adapter = FlowMapAdapter()
        result = adapter.call_tool("flowmap_search", {"query": "rate limit", "limit": 3})

        assert isinstance(result, str)
        assert len(result) > 0

    def test_adapter_call_tool_routes_cat(self):
        """call_tool('flowmap_cat', {...}) routes to server.cat()."""
        adapter = FlowMapAdapter()
        result = adapter.call_tool("flowmap_cat", {
            "repo": "zedbe-aggregator",
            "file": "src/plugins/rateLimit.js"
        })

        assert isinstance(result, str)
        assert "rateLimit" in result or "rate" in result.lower()

    def test_adapter_call_tool_routes_history(self):
        """call_tool('flowmap_history', {...}) routes to server.history()."""
        adapter = FlowMapAdapter()
        result = adapter.call_tool("flowmap_history", {"query": "payment", "limit": 3})

        assert isinstance(result, str)

    def test_adapter_call_tool_routes_repos(self):
        """call_tool('flowmap_repos', {}) routes to server.repos()."""
        adapter = FlowMapAdapter()
        result = adapter.call_tool("flowmap_repos", {})

        assert isinstance(result, str)
        assert "zedbe" in result

    def test_adapter_call_tool_unknown_returns_error(self):
        """call_tool with unknown tool returns error string."""
        adapter = FlowMapAdapter()
        result = adapter.call_tool("flowmap_nonexistent", {})

        assert "Unknown tool" in result or "error" in result.lower()
