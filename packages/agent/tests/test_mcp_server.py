"""Tests for MCP Server - FlowMap tools."""

import pytest


class TestMCPTools:
    """Test that MCP server exposes FlowMap tools."""

    def test_search_tool_exists(self):
        """MCP server exposes flowmap_search tool."""
        from agent.server import FlowMapMCPServer

        server = FlowMapMCPServer()
        tools = server.list_tools()
        tool_names = [t["name"] for t in tools]
        assert "flowmap_search" in tool_names

    def test_search_finds_code(self):
        """Search returns code matching the query."""
        from agent.server import FlowMapMCPServer

        server = FlowMapMCPServer()
        results = server.search("GoogleMapsService")
        assert len(results) > 0
        # Results are structured objects
        assert hasattr(results[0], "file")
        assert hasattr(results[0], "score")

    def test_cat_reads_file(self):
        """Cat returns content of a specific file."""
        from agent.server import FlowMapMCPServer

        server = FlowMapMCPServer()
        content = server.cat("zedbe-map-ms", "src/app.service.ts")
        assert isinstance(content, str)
        assert len(content) > 0

    def test_history_shows_commits(self):
        """History shows commits related to a query."""
        from agent.server import FlowMapMCPServer

        server = FlowMapMCPServer()
        commits = server.history("GoogleMapsService")
        assert isinstance(commits, list)
        # May or may not find commits depending on query
        if len(commits) > 0:
            assert hasattr(commits[0], "sha")
            assert hasattr(commits[0], "message")

    def test_map_shows_structure(self):
        """Map shows repo structure."""
        from agent.server import FlowMapMCPServer

        server = FlowMapMCPServer()
        structure = server.map("payment-service")
        assert isinstance(structure, str)
        assert len(structure) > 0


class TestMCPErrors:
    """Test error handling."""

    def test_unknown_tool_raises_error(self):
        """Unknown tool raises ValueError."""
        from agent.server import FlowMapMCPServer

        server = FlowMapMCPServer()
        with pytest.raises(ValueError, match="Unknown tool"):
            server.call_tool("flowmap_nonexistent", {})

    def test_cat_missing_repo_raises_error(self):
        """Missing repo raises FileNotFoundError."""
        from agent.server import FlowMapMCPServer

        server = FlowMapMCPServer()
        with pytest.raises(FileNotFoundError, match="Repository"):
            server.cat("nonexistent-repo", "file.py")

    def test_cat_missing_file_raises_error(self):
        """Missing file raises FileNotFoundError."""
        from agent.server import FlowMapMCPServer

        server = FlowMapMCPServer()
        with pytest.raises(FileNotFoundError, match="File"):
            server.cat("zedbe-map-ms", "nonexistent.py")


class TestMCPIntegration:
    """Test real FlowMap integration."""

    def test_search_returns_structured_results(self):
        """Search returns structured results, not strings."""
        from agent.server import FlowMapMCPServer

        server = FlowMapMCPServer()
        results = server.search("GoogleMapsService")
        # Real results should be objects with attributes, not strings
        assert len(results) > 0
        first_result = results[0]
        # Should have file and score attributes
        assert hasattr(first_result, "file")
        assert hasattr(first_result, "score")
        assert first_result.score > 0

    def test_cat_returns_real_code(self):
        """Cat returns actual code content."""
        from agent.server import FlowMapMCPServer

        server = FlowMapMCPServer()
        # Use a real repo and file from the config
        content = server.cat("zedbe-map-ms", "src/app.service.ts")
        # Real code has content
        assert len(content) > 0

    def test_history_returns_commit_objects(self):
        """History returns commit objects with sha and message."""
        from agent.server import FlowMapMCPServer

        server = FlowMapMCPServer()
        commits = server.history("GoogleMapsService")
        # Should find some commits
        assert len(commits) > 0
        first_commit = commits[0]
        # Real commits have sha and message
        assert hasattr(first_commit, "sha")
        assert hasattr(first_commit, "message")
        assert len(first_commit.sha) > 0
