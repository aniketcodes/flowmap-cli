"""Slack MCP Client - Connects to external Slack MCP server for hackathon compliance."""

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SlackMCPClient:
    """Client for connecting to Slack MCP server via SSE."""
    
    def __init__(self, server_url: str = "http://localhost:13080/sse"):
        self.server_url = server_url
        self._session = None
        self._read_stream = None
        self._write_stream = None
        self._tools_cache = None
    
    async def connect(self) -> bool:
        """Connect to the Slack MCP server."""
        try:
            from mcp.client.sse import sse_client
            from mcp import ClientSession
            
            # Create SSE connection
            read_stream, write_stream = await sse_client(self.server_url).__aenter__()
            self._read_stream = read_stream
            self._write_stream = write_stream
            
            # Create session
            self._session = ClientSession(read_stream, write_stream)
            await self._session.__aenter__()
            
            # Initialize
            await self._session.initialize()
            
            # List available tools
            result = await self._session.list_tools()
            self._tools_cache = {tool.name: tool for tool in result.tools}
            
            logger.info(f"Connected to Slack MCP server at {self.server_url}")
            logger.info(f"Available tools: {list(self._tools_cache.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Slack MCP server: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the Slack MCP server."""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._read_stream:
            self._read_stream = None
        if self._write_stream:
            self._write_stream = None
    
    def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from the Slack MCP server."""
        if not self._tools_cache:
            return []
        
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in self._tools_cache.values()
        ]
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the Slack MCP server."""
        if not self._session:
            return "Error: Not connected to Slack MCP server"
        
        try:
            result = await self._session.call_tool(tool_name, arguments)
            
            # Extract text content
            if hasattr(result, 'content') and result.content:
                return result.content
            return str(result)
            
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return f"Error calling tool {tool_name}: {e}"


def get_slack_mcp_tools_for_agent() -> list[dict[str, Any]]:
    """Get Slack MCP tools formatted for the agent's tool definitions."""
    client = SlackMCPClient()
    
    try:
        asyncio.get_event_loop().run_until_complete(client.connect())
        tools = client.list_tools()
        
        # Format for agent
        agent_tools = []
        for tool in tools:
            agent_tools.append({
                "name": f"slack_mcp_{tool['name']}",  # Prefix to avoid conflicts
                "description": f"[Slack MCP] {tool['description']}",
                "parameters": tool.get("input_schema", {})
            })
        
        return agent_tools
        
    except Exception as e:
        logger.error(f"Failed to get Slack MCP tools: {e}")
        return []
