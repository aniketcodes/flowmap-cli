"""MCP Adapter — Base class for all MCP server integrations.

Each adapter wraps a specific MCP server (FlowMap, Slack, Grafana, etc.)
and exposes a uniform interface for the Agent to use.
"""

from abc import ABC, abstractmethod
from typing import Any


class MCPAdapter(ABC):
    """Base adapter for all MCP server integrations.

    Subclasses must implement:
    - name(): Unique prefix for tool names
    - list_tools(): Return MCP-compliant tool definitions
    - call_tool(): Execute a tool and return string result

    Optional overrides:
    - connect(): Establish connection to MCP server
    - disconnect(): Clean up connection resources
    """

    @abstractmethod
    def name(self) -> str:
        """Unique prefix for tool names (e.g., 'flowmap', 'slack', 'grafana').

        Used to disambiguate tools when multiple adapters are loaded.
        Example: flowmap_search, slack_conversations_history, grafana_query
        """
        pass

    @abstractmethod
    def list_tools(self) -> list[dict]:
        """Return MCP-compliant tool definitions.

        Each tool must have:
        - name: str
        - description: str
        - inputSchema: dict (JSON Schema for parameters)

        Returns:
            list[dict]: Tool definitions in MCP format
        """
        pass

    @abstractmethod
    def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return the result as a string.

        Args:
            name: Tool name
            args: Tool arguments as a dictionary

        Returns:
            str: Tool result formatted as a string
        """
        pass

    def connect(self) -> bool:
        """Establish connection to the MCP server.

        Override if your adapter needs explicit connection setup.
        Default returns True (no connection needed).

        Returns:
            bool: True if connected successfully
        """
        return True

    def disconnect(self):
        """Clean up connection resources.

        Override if your adapter needs explicit cleanup.
        Default is a no-op.
        """
        pass
