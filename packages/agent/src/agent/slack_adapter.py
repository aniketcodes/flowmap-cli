"""Slack Adapter — Wraps SlackMCPClient with MCPAdapter interface.

Uses a dedicated event loop thread for all async MCP operations.
This avoids the "loop already running" crash and keeps sessions bound
to the correct event loop.
"""

import asyncio
import logging
import threading
from typing import Any

from .adapter import MCPAdapter
from .slack_mcp_client import SlackMCPClient

logger = logging.getLogger(__name__)

# Lazy-initialized event loop for async MCP operations
_async_loop = None
_async_thread = None
_loop_lock = threading.Lock()


def _get_async_loop() -> asyncio.AbstractEventLoop:
    """Get or create the dedicated async event loop (lazy init)."""
    global _async_loop, _async_thread
    if _async_loop is None:
        with _loop_lock:
            if _async_loop is None:  # Double-check after acquiring lock
                _async_loop = asyncio.new_event_loop()
                _async_thread = threading.Thread(
                    target=_async_loop.run_forever,
                    daemon=True
                )
                _async_thread.start()
    return _async_loop


class SlackAdapter(MCPAdapter):
    """Adapter for Slack MCP server.

    Wraps SlackMCPClient and exposes it through the standard MCPAdapter
    interface. Tool names are automatically prefixed with 'slack_'.

    Uses run_coroutine_threadsafe() to dispatch async calls to the
    dedicated event loop thread, avoiding event loop binding issues.
    """

    def __init__(self, client: SlackMCPClient = None, url: str = None):
        self._url = url or "http://localhost:13080/sse"
        self._client = client or SlackMCPClient(server_url=self._url)
        self._connected = False

    def name(self) -> str:
        return "slack"

    def list_tools(self) -> list[dict]:
        if not self._connected:
            return []

        tools = self._client.list_tools()
        # Prefix tool names to avoid conflicts (copy to avoid mutating cache)
        return [
            {**tool, "name": f"slack_{tool['name']}"}
            for tool in tools
        ]

    def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Route tool call to Slack MCP server."""
        if not self._connected:
            return "Error: Not connected to Slack MCP server"

        # Remove prefix
        mcp_name = name.removeprefix("slack_")

        try:
            # Dispatch to dedicated async loop — avoids "loop already running"
            # and keeps session bound to the correct event loop
            loop = _get_async_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._client.call_tool(mcp_name, args),
                loop
            )
            return future.result(timeout=30)
        except Exception as e:
            logger.error("Slack MCP tool error: %s %s", mcp_name, e, exc_info=True)
            return f"Error calling {name}: {e}"

    def connect(self) -> bool:
        """Connect to Slack MCP server."""
        try:
            # Dispatch to dedicated async loop
            loop = _get_async_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._client.connect(),
                loop
            )
            self._connected = future.result(timeout=10)

            if self._connected:
                logger.info("Connected to Slack MCP server at %s", self._url)
            else:
                logger.warning("Failed to connect to Slack MCP server")

            return self._connected
        except Exception as e:
            logger.error("Slack MCP connection error: %s", e, exc_info=True)
            return False

    def disconnect(self):
        """Disconnect from Slack MCP server."""
        if self._client and self._connected:
            try:
                loop = _get_async_loop()
                future = asyncio.run_coroutine_threadsafe(
                    self._client.disconnect(),
                    loop
                )
                future.result(timeout=5)
            except Exception as e:
                logger.error("Slack MCP disconnect error: %s", e)
            finally:
                self._connected = False
