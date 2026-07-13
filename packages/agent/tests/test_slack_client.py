"""Tests for SlackClient - Slack MCP integration."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSlackClient:
    """Test SlackClient behaviors."""

    def test_slack_client_can_get_channel_history(self):
        """SlackClient can retrieve messages from a channel."""
        from agent.slack_client import SlackClient, SlackMessage
        
        # Arrange
        mock_app = Mock()
        mock_app.client.conversations_history.return_value = {
            "ok": True,
            "messages": [
                {
                    "ts": "1234567890.123456",
                    "user": "U123456",
                    "text": "We're seeing 402 errors on the payment endpoint",
                    "channel": "C123456"
                },
                {
                    "ts": "1234567890.123457",
                    "user": "U789012",
                    "text": "I think it's related to the rate limiter change",
                    "channel": "C123456"
                }
            ]
        }
        
        client = SlackClient(app=mock_app)
        
        # Act
        messages = client.get_channel_history("C123456", limit=10)
        
        # Assert
        assert len(messages) == 2
        assert messages[0].text == "We're seeing 402 errors on the payment endpoint"
        assert messages[0].user == "U123456"
        assert messages[0].ts == "1234567890.123456"
        assert messages[1].text == "I think it's related to the rate limiter change"

    def test_slack_client_can_get_thread_replies(self):
        """SlackClient can retrieve replies in a thread."""
        from agent.slack_client import SlackClient, SlackMessage
        
        # Arrange
        mock_app = Mock()
        mock_app.client.conversations_replies.return_value = {
            "ok": True,
            "messages": [
                {
                    "ts": "1234567890.123456",
                    "user": "U123456",
                    "text": "We're seeing 402 errors on the payment endpoint",
                    "channel": "C123456"
                },
                {
                    "ts": "1234567890.123457",
                    "user": "U789012",
                    "text": "I think it's related to the rate limiter change",
                    "channel": "C123456",
                    "thread_ts": "1234567890.123456"
                },
                {
                    "ts": "1234567890.123458",
                    "user": "U345678",
                    "text": "Confirmed - we disabled the rate limiter and errors stopped",
                    "channel": "C123456",
                    "thread_ts": "1234567890.123456"
                }
            ]
        }
        
        client = SlackClient(app=mock_app)
        
        # Act
        messages = client.get_thread_replies("C123456", "1234567890.123456")
        
        # Assert
        assert len(messages) == 3
        assert messages[0].text == "We're seeing 402 errors on the payment endpoint"
        assert messages[1].text == "I think it's related to the rate limiter change"
        assert messages[2].text == "Confirmed - we disabled the rate limiter and errors stopped"

    def test_slack_client_can_list_channels(self):
        """SlackClient can list available channels."""
        from agent.slack_client import SlackClient, SlackChannel
        
        # Arrange
        mock_app = Mock()
        mock_app.client.conversations_list.return_value = {
            "ok": True,
            "channels": [
                {"id": "C123456", "name": "incidents"},
                {"id": "C789012", "name": "engineering"},
                {"id": "C345678", "name": "general"}
            ]
        }
        
        client = SlackClient(app=mock_app)
        
        # Act
        channels = client.list_channels()
        
        # Assert
        assert len(channels) == 3
        assert channels[0].name == "incidents"
        assert channels[1].name == "engineering"
        assert channels[2].name == "general"

    def test_slack_client_handles_api_errors(self):
        """SlackClient handles Slack API errors gracefully."""
        from agent.slack_client import SlackClient
        
        # Arrange
        mock_app = Mock()
        mock_app.client.conversations_history.return_value = {
            "ok": False,
            "error": "not_in_channel"
        }
        
        client = SlackClient(app=mock_app)
        
        # Act
        messages = client.get_channel_history("C123456", limit=10)
        
        # Assert
        assert messages == []


class TestAgentSlackIntegration:
    """Test Agent can call Slack tools alongside FlowMap tools."""

    def test_agent_can_call_slack_history_tool(self):
        """Agent can call slack_history tool to get channel messages."""
        from agent.agent import Agent
        from agent.llm import LLMResponse, ToolCall
        from agent.slack_client import SlackClient
        from agent.server import FlowMapMCPServer

        # Arrange
        mock_mcp = Mock()
        mock_mcp.list_tools.return_value = FlowMapMCPServer.list_tools(FlowMapMCPServer.__new__(FlowMapMCPServer))

        mock_app = Mock()
        mock_app.client.conversations_history.return_value = {
            "ok": True,
            "messages": [
                {"ts": "123", "user": "U1", "text": "Payment 402 errors in prod", "channel": "C1"}
            ]
        }

        slack_client = SlackClient(app=mock_app)
        mock_llm = Mock()

        # LLM makes 2 tool calls then returns text (satisfies MIN_TOOL_CALLS=2)
        mock_llm.chat.side_effect = [
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc1", name="flowmap_search", arguments={"query": "402 error"}),
            ]),
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc2", name="slack_history", arguments={"channel_id": "C1", "limit": 10}),
            ]),
            LLMResponse(content="Found payment 402 errors in Slack channel.", tool_calls=None),
        ]

        agent = Agent(mcp=mock_mcp, llm=mock_llm, slack_client=slack_client)

        # Act
        result = agent.diagnose("Why are we getting 402 errors?")

        # Assert
        assert result is not None
        assert len(result) > 0
        assert "payment 402" in result.lower() or "402" in result

    def test_agent_tool_definitions_include_slack_tools(self):
        """Agent includes slack tools in its tool definitions."""
        from agent.agent import Agent
        from agent.slack_client import SlackClient
        from agent.server import FlowMapMCPServer

        # Arrange
        mock_mcp = Mock()
        mock_mcp.list_tools.return_value = FlowMapMCPServer.list_tools(FlowMapMCPServer.__new__(FlowMapMCPServer))

        mock_llm = Mock()
        mock_app = Mock()
        slack_client = SlackClient(app=mock_app)

        agent = Agent(mcp=mock_mcp, llm=mock_llm, slack_client=slack_client)

        # Act
        tool_defs = agent._build_openai_tools()
        tool_names = [t["function"]["name"] for t in tool_defs if t.get("type") == "function"]

        # Assert
        assert "slack_history" in tool_names
        assert "slack_search" in tool_names
        assert "flowmap_search" in tool_names
        assert "flowmap_cat" in tool_names


class TestMCPServerCompliance:
    """Test FlowMap MCP server for hackathon compliance."""

    def test_mcp_server_creates_successfully(self):
        """FlowMap MCP server can be created."""
        from agent.mcp_transport import create_flowmap_mcp_server
        
        mcp = create_flowmap_mcp_server()
        assert mcp is not None
        assert mcp.name == "FlowMap Code Intelligence"

    def test_mcp_server_has_tools(self):
        """FlowMap MCP server exposes tools."""
        from agent.mcp_transport import create_flowmap_mcp_server
        
        mcp = create_flowmap_mcp_server()
        
        # Check that tools are registered
        assert hasattr(mcp, '_tool_manager')
        assert mcp._tool_manager is not None


class TestSlackMCPClient:
    """Test Slack MCP Client for hackathon compliance."""

    def test_slack_mcp_client_can_be_created(self):
        """SlackMCPClient can be instantiated."""
        from agent.slack_mcp_client import SlackMCPClient
        
        client = SlackMCPClient(server_url="http://localhost:3000/sse")
        assert client is not None
        assert client.server_url == "http://localhost:3000/sse"

    def test_slack_mcp_client_has_list_tools_method(self):
        """SlackMCPClient has list_tools method."""
        from agent.slack_mcp_client import SlackMCPClient
        
        client = SlackMCPClient()
        assert hasattr(client, 'list_tools')
        assert callable(client.list_tools)

    def test_slack_mcp_client_has_call_tool_method(self):
        """SlackMCPClient has call_tool method."""
        from agent.slack_mcp_client import SlackMCPClient
        
        client = SlackMCPClient()
        assert hasattr(client, 'call_tool')
        assert callable(client.call_tool)

    def test_agent_includes_slack_mcp_tools_when_connected(self):
        """Agent includes slack tools from SlackAdapter."""
        from agent.agent import Agent
        from agent.slack_adapter import SlackAdapter
        from agent.server import FlowMapMCPServer

        # Arrange - use adapters directly
        flowmap_adapter = Mock()
        flowmap_adapter.name.return_value = "flowmap"
        flowmap_adapter.list_tools.return_value = FlowMapMCPServer.list_tools(FlowMapMCPServer.__new__(FlowMapMCPServer))

        slack_adapter = Mock()
        slack_adapter.name.return_value = "slack"
        slack_adapter.list_tools.return_value = [
            {
                "name": "slack_conversations_history",
                "description": "Get channel history",
                "inputSchema": {"type": "object", "properties": {"channel_id": {"type": "string"}}, "required": ["channel_id"]},
            },
            {
                "name": "slack_conversations_search_messages",
                "description": "Search messages",
                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
        ]

        mock_llm = Mock()

        agent = Agent(adapters=[flowmap_adapter, slack_adapter], llm=mock_llm)

        # Act
        tool_defs = agent._build_openai_tools()
        tool_names = [t["function"]["name"] for t in tool_defs if t.get("type") == "function"]

        # Assert
        assert "slack_conversations_history" in tool_names
        assert "slack_conversations_search_messages" in tool_names
        assert "flowmap_search" in tool_names
