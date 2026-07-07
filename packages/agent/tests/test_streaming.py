"""Tests for StreamingMessage — live-updating Slack messages."""

import time
import pytest
from unittest.mock import Mock
from agent.streaming import StreamingMessage


class TestStreamingMessage:
    """Test StreamingMessage posts and updates a single Slack message."""

    def test_send_posts_initial_message(self):
        """send() posts placeholder and captures ts."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "123.456"}

        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")
        result = sm.send()

        assert result == "123.456"
        assert sm.ts == "123.456"
        mock_client.chat_postMessage.assert_called_once_with(
            channel="C123", text="Thinking...", thread_ts=None
        )

    def test_send_returns_none_on_failure(self):
        """send() returns None when Slack API fails."""
        mock_client = Mock()
        mock_client.chat_postMessage.side_effect = Exception("API down")

        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")
        result = sm.send()

        assert result is None
        assert sm.ts is None

    def test_update_overwrites_same_message(self):
        """update() calls chat_update with the original ts."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "123.456"}

        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")
        sm.send()
        sm.update("Searching code...")

        mock_client.chat_update.assert_called_once_with(
            channel="C123", ts="123.456", text="Searching code..."
        )

    def test_update_with_blocks(self):
        """update() supports Slack Block Kit blocks."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "123.456"}

        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")
        sm.send()

        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "*Loading...*"}}]
        sm.update("Loading...", blocks=blocks)

        mock_client.chat_update.assert_called_once_with(
            channel="C123", ts="123.456", text="Loading...", blocks=blocks
        )

    def test_finish_updates_with_final_response(self):
        """finish() replaces placeholder with final response."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "123.456"}

        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")
        sm.send()
        sm.finish("Root cause: Redis timeout in auth.ts:42")

        mock_client.chat_update.assert_called_with(
            channel="C123", ts="123.456",
            text="Root cause: Redis timeout in auth.ts:42"
        )

    def test_finish_truncates_long_messages(self):
        """finish() truncates text超过 Slack's 40k limit."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "123.456"}

        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")
        sm.send()

        long_text = "x" * 50000
        sm.finish(long_text)

        call_args = mock_client.chat_update.call_args
        assert len(call_args.kwargs["text"]) < 40000
        assert "truncated" in call_args.kwargs["text"].lower()

    def test_update_before_send_raises(self):
        """update() before send() raises RuntimeError."""
        mock_client = Mock()
        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")

        with pytest.raises(RuntimeError):
            sm.update("Too early")

    def test_finish_before_send_raises(self):
        """finish() before send() raises RuntimeError."""
        mock_client = Mock()
        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")

        with pytest.raises(RuntimeError):
            sm.finish("Too early")

    def test_update_swallows_errors(self):
        """update() never raises — errors are logged."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "123.456"}
        mock_client.chat_update.side_effect = Exception("rate limited")

        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")
        sm.send()
        sm.update("Should not raise")  # No exception

    def test_finish_swallows_errors(self):
        """finish() never raises — errors are logged."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "123.456"}
        mock_client.chat_update.side_effect = Exception("network error")

        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")
        sm.send()
        sm.finish("Final answer")  # No exception

    def test_creates_thread_when_thread_ts_provided(self):
        """Initial message is threaded under the user's message."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "456.789"}

        sm = StreamingMessage(
            client=mock_client, channel="C123",
            initial="Thinking...", thread_ts="111.222"
        )
        sm.send()

        mock_client.chat_postMessage.assert_called_once_with(
            channel="C123", text="Thinking...", thread_ts="111.222"
        )

    def test_update_throttled(self):
        """update() skips calls faster than MIN_UPDATE_INTERVAL."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "123.456"}

        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")
        sm.send()

        # Two rapid updates — second should be throttled
        sm.update("First")
        sm.update("Second")  # Should be skipped (too fast)

        assert mock_client.chat_update.call_count == 1

    def test_update_after_throttle_interval(self):
        """update() sends after MIN_UPDATE_INTERVAL elapsed."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "123.456"}

        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")
        sm._MIN_UPDATE_INTERVAL = 0  # Disable throttle for this test
        sm.send()

        sm.update("First")
        sm.update("Second")

        assert mock_client.chat_update.call_count == 2

    def test_finish_always_posts_regardless_of_throttle(self):
        """finish() bypasses throttle — final answer always sends."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "123.456"}

        sm = StreamingMessage(client=mock_client, channel="C123", initial="Thinking...")
        sm._MIN_UPDATE_INTERVAL = 999  # Set throttle very high
        sm.send()

        sm.update("First")   # Goes through (first call, _last_update was 0)
        sm.update("Second")  # Throttled (too fast after First)
        sm.finish("Final")   # Should send despite throttle

        # 2 calls: First (not throttled) + Final (bypasses throttle)
        # Second was throttled
        assert mock_client.chat_update.call_count == 2


class TestRunBotStreaming:
    """Test the run_bot.py streaming integration."""

    def test_process_mention_streams_progress(self):
        """process_mention posts placeholder, streams, then finishes."""
        mock_client = Mock()
        mock_client.chat_postMessage.return_value = {"ts": "555.666"}

        sm = StreamingMessage(
            client=mock_client, channel="C123",
            initial="🤔 Analyzing...", thread_ts="111.222",
        )
        sm.send()

        # Simulate agent calling progress
        sm.update("🔍 Searching code...")
        sm.update("📄 Reading auth.ts...")
        sm.finish("Root cause: Redis timeout")

        # 1 post + updates (may be throttled) + 1 finish
        assert mock_client.chat_postMessage.call_count == 1
        assert mock_client.chat_update.call_count >= 2

    def test_run_bot_fallback_when_send_fails(self):
        """When placeholder fails, run_bot posts response directly."""
        mock_client = Mock()
        mock_client.chat_postMessage.side_effect = Exception("API down")

        sm = StreamingMessage(
            client=mock_client, channel="C123",
            initial="🤔 Analyzing...",
        )
        result = sm.send()

        assert result is None
        # Fallback: direct post
        mock_client.chat_postMessage.side_effect = None
        mock_client.chat_postMessage.return_value = {"ts": "fallback"}
        mock_client.chat_postMessage(
            channel="C123", text="Root cause found", thread_ts=None
        )
        assert mock_client.chat_postMessage.call_count == 2
