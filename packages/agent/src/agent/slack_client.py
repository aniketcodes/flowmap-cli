"""SlackClient - Wraps Slack API for conversation search."""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SlackMessage:
    """Represents a Slack message."""
    ts: str
    user: str
    text: str
    channel: str
    thread_ts: Optional[str] = None
    reply_count: Optional[int] = None


@dataclass
class SlackChannel:
    """Represents a Slack channel."""
    id: str
    name: str


class SlackClient:
    """Wraps Slack API for conversation search and retrieval."""

    def __init__(self, app=None):
        """Initialize with a Bolt App instance.
        
        Args:
            app: slack_bolt.App instance (if None, creates from env)
        """
        if app is None:
            from slack_bolt import App
            import os
            app = App(
                token=os.getenv("SLACK_BOT_TOKEN"),
                signing_secret=os.getenv("SLACK_SIGNING_SECRET")
            )
        self._app = app

    def get_channel_history(self, channel_id: str, limit: int = 100) -> list[SlackMessage]:
        """Retrieve messages from a channel.
        
        Args:
            channel_id: Slack channel ID (e.g., 'C123456')
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of SlackMessage objects
        """
        try:
            result = self._app.client.conversations_history(
                channel=channel_id,
                limit=limit
            )
            
            if not result.get("ok"):
                logger.warning("conversations_history failed: %s", result.get("error"))
                return []
            
            messages = []
            for msg in result.get("messages", []):
                messages.append(SlackMessage(
                    ts=msg.get("ts", ""),
                    user=msg.get("user", ""),
                    text=msg.get("text", ""),
                    channel=channel_id,
                    thread_ts=msg.get("thread_ts"),
                    reply_count=msg.get("reply_count")
                ))
            
            return messages
            
        except Exception as e:
            logger.error("get_channel_history error: %s", e)
            return []

    def get_thread_replies(self, channel_id: str, thread_ts: str) -> list[SlackMessage]:
        """Retrieve replies in a thread.
        
        Args:
            channel_id: Slack channel ID
            thread_ts: Timestamp of the parent message
            
        Returns:
            List of SlackMessage objects (parent + replies)
        """
        try:
            result = self._app.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts
            )
            
            if not result.get("ok"):
                logger.warning("conversations_replies failed: %s", result.get("error"))
                return []
            
            messages = []
            for msg in result.get("messages", []):
                messages.append(SlackMessage(
                    ts=msg.get("ts", ""),
                    user=msg.get("user", ""),
                    text=msg.get("text", ""),
                    channel=channel_id,
                    thread_ts=msg.get("thread_ts"),
                    reply_count=msg.get("reply_count")
                ))
            
            return messages
            
        except Exception as e:
            logger.error("get_thread_replies error: %s", e)
            return []

    def list_channels(self) -> list[SlackChannel]:
        """List all channels the bot has access to.
        
        Returns:
            List of SlackChannel objects
        """
        try:
            result = self._app.client.conversations_list(
                types="public_channel,private_channel"
            )
            
            if not result.get("ok"):
                logger.warning("conversations_list failed: %s", result.get("error"))
                return []
            
            channels = []
            for ch in result.get("channels", []):
                channels.append(SlackChannel(
                    id=ch.get("id", ""),
                    name=ch.get("name", "")
                ))
            
            return channels
            
        except Exception as e:
            logger.error("list_channels error: %s", e)
            return []

    def search_messages(self, query: str, channel_id: str = None, limit: int = 10) -> list[SlackMessage]:
        """Search for messages matching a query.
        
        Note: This uses conversations_history with basic filtering.
        For full search, use Slack's search API with user token.
        
        Args:
            query: Search query (will match against message text)
            channel_id: Optional channel ID to search in
            limit: Maximum number of results
            
        Returns:
            List of SlackMessage objects
        """
        # If no channel specified, search all channels bot is in
        if channel_id is None:
            channels = self.list_channels()
            all_messages = []
            for ch in channels:
                messages = self.get_channel_history(ch.id, limit=50)
                all_messages.extend(messages)
        else:
            all_messages = self.get_channel_history(channel_id, limit=100)
        
        # Filter by query (simple text matching)
        query_lower = query.lower()
        matches = [
            msg for msg in all_messages
            if query_lower in msg.text.lower()
        ]
        
        return matches[:limit]
