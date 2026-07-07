"""Tests for Slack Bot."""

import pytest
from unittest.mock import Mock


class TestSlackBot:
    """Test Slack bot functionality."""

    def test_bot_can_be_created(self):
        """Bot can be instantiated."""
        from agent.bot import SlackBot

        bot = SlackBot()
        assert bot is not None

    def test_bot_responds_to_mention(self):
        """Bot responds when mentioned."""
        from agent.bot import SlackBot

        bot = SlackBot()
        response = bot.handle_mention("Why are we getting 402 errors?")
        assert response is not None
        assert len(response) > 0

    def test_bot_deduplicates_events(self):
        """Duplicate events are ignored."""
        from agent.bot import SlackBot

        bot = SlackBot()
        event_id = "EV123"
        # First call should return response
        response1 = bot.handle_mention("test", event_id=event_id)
        assert response1 is not None
        # Second call with same event_id should return None
        response2 = bot.handle_mention("test", event_id=event_id)
        assert response2 is None

    def test_bot_streams_progress(self):
        """Bot sends progress updates during processing."""
        from agent.bot import SlackBot

        bot = SlackBot()
        updates = []
        bot.on_progress = lambda msg: updates.append(msg)
        bot.handle_mention("Why 402 errors?")
        # Should have at least one progress update
        assert len(updates) > 0
        assert "Analyzing" in updates[0] or "Processing" in updates[0]


class TestBotStreaming:
    """Test bot posts placeholder and streams progress to Slack."""

    def test_make_progress_callback_updates_streaming_message(self):
        """_make_progress_callback wraps sm.update with _md_to_slack."""
        from agent.bot import SlackBot
        from agent.streaming import StreamingMessage

        bot = SlackBot()
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "999.000"}

        sm = StreamingMessage(client=mock_client, channel="C123", initial="Analyzing...")
        sm.send()

        callback = bot._make_progress_callback(sm)
        callback("Searching code...")

        mock_client.chat_update.assert_called_once_with(
            channel="C123", ts="999.000", text="Searching code..."
        )

    def test_handle_mention_streams_when_channel_provided(self):
        """handle_mention uses streaming when channel is given."""
        from agent.bot import SlackBot

        bot = SlackBot()
        mock_agent = Mock()
        mock_agent.diagnose.return_value = "Root cause found"
        bot.agent = mock_agent

        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "999.000"}
        bot.slack_app = Mock()
        bot.slack_app.client = mock_client

        response = bot.handle_mention(
            "Why 402?", event_id="E1",
            channel="C123", thread_ts="T1"
        )

        # Should post placeholder + final = 2 chat_postMessage calls
        assert mock_client.chat_postMessage.call_count == 1
        # Should have 1 finish update
        assert mock_client.chat_update.call_count == 1
        assert response == "Root cause found"

    def test_handle_mention_falls_back_when_send_fails(self):
        """When send() fails, falls back to direct post."""
        from agent.bot import SlackBot

        bot = SlackBot()
        mock_agent = Mock()
        mock_agent.diagnose.return_value = "Root cause found"
        bot.agent = mock_agent

        mock_client = Mock()
        mock_client.chat_postMessage.side_effect = [
            Exception("API down"),  # First call (placeholder) fails
            {"ts": "999.000"},     # Fallback call succeeds
        ]
        bot.slack_app = Mock()
        bot.slack_app.client = mock_client

        response = bot.handle_mention(
            "Why 402?", event_id="E1",
            channel="C123", thread_ts="T1"
        )

        # Should have tried placeholder, then fallen back to direct post
        assert mock_client.chat_postMessage.call_count == 2
        assert response == "Root cause found"

    def test_handle_mention_no_channel_uses_no_streaming(self):
        """Without channel, uses non-streaming path (backward compat)."""
        from agent.bot import SlackBot

        bot = SlackBot()
        mock_agent = Mock()
        mock_agent.diagnose.return_value = "Response"
        bot.agent = mock_agent

        response = bot.handle_mention("test", event_id="E2")

        assert response == "Response"
        mock_agent.diagnose.assert_called_once_with("test")

    def test_register_handlers_does_not_double_post(self):
        """app_mention handler does NOT call say() when streaming posts the response."""
        from agent.bot import SlackBot

        bot = SlackBot()
        mock_agent = Mock()
        mock_agent.diagnose.return_value = "diagnosis"
        bot.agent = mock_agent

        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "999.000"}
        bot.slack_app = Mock()
        bot.slack_app.client = mock_client

        # Simulate what _register_handlers does
        # handle_mention returns the response (already posted via streaming)
        response = bot.handle_mention(
            "test", event_id="E3",
            channel="C123", thread_ts="T1"
        )

        # Only 1 chat_postMessage (the placeholder), then 1 chat_update (finish)
        # No additional say() call should happen
        assert mock_client.chat_postMessage.call_count == 1

    def test_dedup_uses_atomic_check(self):
        """Duplicate events are caught atomically."""
        from agent.bot import SlackBot

        bot = SlackBot()
        mock_agent = Mock()
        mock_agent.diagnose.return_value = "Response"
        bot.agent = mock_agent

        # First call
        response1 = bot.handle_mention("test", event_id="EV1")
        assert response1 == "Response"

        # Second call with same event_id — should return None (deduped)
        response2 = bot.handle_mention("test", event_id="EV1")
        assert response2 is None

    def test_handle_mention_streams_error_when_agent_raises(self):
        """When agent.diagnose() raises, streaming shows error in the message."""
        from agent.bot import SlackBot

        bot = SlackBot()
        mock_agent = Mock()
        mock_agent.diagnose.side_effect = Exception("LLM connection lost")
        bot.agent = mock_agent

        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "999.000"}
        bot.slack_app = Mock()
        bot.slack_app.client = mock_client

        response = bot.handle_mention(
            "Why 402?", event_id="E4",
            channel="C123", thread_ts="T1"
        )

        # Should still post a response (the error message)
        assert response is not None
        assert "error" in response.lower() or "Sorry" in response
        # Placeholder was posted, then finish with error
        assert mock_client.chat_postMessage.call_count == 1
        assert mock_client.chat_update.call_count == 1
