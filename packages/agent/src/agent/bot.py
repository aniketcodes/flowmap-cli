"""Slack Bot - Handles Slack events and responds to mentions with streaming."""

import os
import re
import logging
import time
from typing import Callable, Optional
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from agent.security import TTLCache
from agent.actions import ActionRegistry, ActionPlanner, ActionExecutor, build_action_blocks

logger = logging.getLogger(__name__)

# Import _md_to_slack from bot module (used by ActionExecutor)
# Avoid circular import by defining it here and exporting
def _md_to_slack(text: str) -> str:
    """Convert markdown to Slack mrkdwn format."""
    text = text.replace("→", "->")
    text = re.sub(r"```[\w]*\n(.*?)```", r"```\1```", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.*?)\*\*", r"*\1*", text)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*[-*]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    text = re.sub(r"^_{1,8}(.*?)_{1,8}$", r"\1", text, flags=re.MULTILINE)
    return text.strip()


class SlackBot:
    """Slack bot that responds to mentions with streaming."""

    def __init__(self, agent=None, app_token: str = None, bot_token: str = None):
        self._seen_events = TTLCache()
        self.on_progress: Optional[Callable[[str], None]] = None
        self.agent = agent

        self.app_token = app_token or os.getenv("SLACK_APP_TOKEN")
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.signing_secret = os.getenv("SLACK_SIGNING_SECRET")

        # Action planner components
        llm = getattr(agent, 'llm', None) if agent else None
        self._action_planner = ActionPlanner(llm=llm)
        self._action_registry = ActionRegistry(ttl=3600)
        self._action_executor = ActionExecutor(client=None)

        if self.bot_token:
            self.slack_app = App(token=self.bot_token)
            self._action_executor.client = self.slack_app.client
            self._register_handlers()
        else:
            self.slack_app = None
            logger.warning("Slack tokens not provided, running in standalone mode")

    def _register_handlers(self):
        @self.slack_app.event("app_mention")
        def handle_mention_event(event, say):
            message = event.get("text", "")
            event_id = event.get("ts", "")
            channel = event.get("channel", "")
            thread_ts = event.get("thread_ts", event_id)

            for word in message.split():
                if word.startswith("<@") and word.endswith(">"):
                    message = message.replace(word, "")
            message = message.strip()

            # Streaming: handle_mention posts the response directly.
            # Do NOT call say() afterward — it would double-post.
            self.handle_mention(
                message, event_id=event_id,
                channel=channel, thread_ts=thread_ts,
            )

        @self.slack_app.action(re.compile(r"^[a-z_]+\|\d+\.\d+$"))
        def handle_action_button(ack, action, body):
            ack()
            action_id = action.get("value")
            message_ts = body.get("container", {}).get("message_ts", "")

            if not action_id or not message_ts:
                logger.warning("Button click missing action_id or message_ts")
                return

            channel_id = body.get("channel", {}).get("id", "")
            diagnosis = self._action_registry.get_diagnosis(message_ts)

            logger.info("action_click action_id=%s message_ts=%s", action_id, message_ts)
            result = self._action_executor.execute(action_id, message_ts, channel_id, diagnosis)
            logger.info("action_result action_id=%s result=%s", action_id, result)

            # Post result as a thread reply
            if self.slack_app:
                try:
                    self.slack_app.client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=message_ts,
                        text=f":white_check_mark: {result}",
                    )
                except Exception as e:
                    logger.error("Failed to post action result: %s", e)

    def _make_progress_callback(self, streaming_msg):
        """Create a progress callback that updates a StreamingMessage."""
        def callback(text: str):
            streaming_msg.update(text)
        return callback

    def handle_mention(
        self,
        message: str,
        event_id: str = None,
        channel: str = None,
        thread_ts: str = None,
    ) -> Optional[str]:
        """Handle a mention. Returns response text, or None if deduped.

        When channel is provided: uses streaming (placeholder -> updates -> finish).
        When channel is None: non-streaming fallback (backward compat).
        """
        # Atomic dedup
        if event_id and self._seen_events.contains_and_add(event_id):
            logger.info("duplicate_event event_id=%s", event_id)
            return None

        # Streaming path (when channel provided)
        if channel and self.slack_app:
            from agent.streaming import StreamingMessage

            sm = StreamingMessage(
                client=self.slack_app.client,
                channel=channel,
                initial="🤔 Analyzing your query...",
                thread_ts=thread_ts,
            )
            placeholder_ts = sm.send()

            # Fallback: if placeholder failed, just get response and post directly
            if not placeholder_ts:
                logger.warning("Placeholder failed, falling back to direct post")
                try:
                    response = self.agent.diagnose(message) if self.agent else f"Processing: {message}"
                except Exception as e:
                    logger.error("mention_error error=%s", e, exc_info=True)
                    response = "Sorry, I encountered an error. Please try again later."
                try:
                    self.slack_app.client.chat_postMessage(
                        channel=channel, text=_md_to_slack(response), thread_ts=thread_ts
                    )
                except Exception as e:
                    logger.error("Fallback post failed: %s", e)
                return response

            # Streaming path
            progress = self._make_progress_callback(sm)
            try:
                response = self.agent.diagnose(message, on_progress=progress) if self.agent else f"Processing: {message}"
            except Exception as e:
                logger.error("mention_error error=%s", e, exc_info=True)
                response = "Sorry, I encountered an error. Please try again later."

            # Build action buttons if planner and registry are available
            blocks = None
            if self._action_planner and self._action_registry:
                actions = self._action_planner.recommend(response)
                if actions:
                    self._action_registry.store(placeholder_ts, actions, diagnosis=response)
                    blocks = build_action_blocks(response, actions, placeholder_ts)

            sm.finish(_md_to_slack(response))

            # Post action buttons as a separate message (chat_update doesn't support interactive elements)
            if blocks and self.slack_app:
                try:
                    self.slack_app.client.chat_postMessage(
                        channel=channel,
                        thread_ts=placeholder_ts,
                        blocks=blocks,
                        text="Suggested actions",
                    )
                except Exception as e:
                    logger.error("Failed to post action buttons: %s", e)

            return response

        # Non-streaming fallback (backward compat — no channel, no slack_app)
        if self.on_progress:
            self.on_progress("Analyzing your query...")
        try:
            return self.agent.diagnose(message) if self.agent else f"Processing: {message}"
        except Exception as e:
            logger.error("mention_error error=%s", e, exc_info=True)
            return "Sorry, I encountered an error. Please try again later."

    def start(self):
        if not self.slack_app:
            logger.error("Cannot start: Slack tokens not provided")
            return

        if self.app_token:
            handler = SocketModeHandler(self.slack_app, self.app_token)
            handler.start()
        else:
            logger.info("No SLACK_APP_TOKEN, running in HTTP mode")
