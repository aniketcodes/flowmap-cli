"""StreamingMessage — live-updating Slack message.

Posts an initial placeholder, then overwrites via chat_update as the agent works.
Thread-safe, throttled, with graceful degradation on failure.
"""

import logging
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Slack text limit (leave headroom)
_MAX_TEXT_LENGTH = 39000


class StreamingMessage:
    """A single Slack message that gets updated in real-time.

    Thread-safe: all state mutations protected by a lock.
    Throttled: updates closer than MIN_UPDATE_INTERVAL are skipped.
    Resilient: update()/finish() never raise on API errors.
               Raises RuntimeError if called before send() (programming error).

    Usage:
        sm = StreamingMessage(client, channel="C123", initial="Thinking...")
        sm.send()                          # posts placeholder
        sm.update("Searching code...")     # overwrites with progress
        sm.finish("Root cause: ...")       # final overwrite with answer
    """

    _MIN_UPDATE_INTERVAL = 0.5  # seconds — responsive but safe under 50/min Slack limit

    def __init__(
        self,
        client,
        channel: str,
        initial: str = "Thinking...",
        thread_ts: Optional[str] = None,
    ):
        self._client = client
        self._channel = channel
        self._initial = initial
        self._thread_ts = thread_ts
        self._lock = threading.Lock()
        self._last_update: float = 0
        self.ts: Optional[str] = None

    def send(self) -> Optional[str]:
        """Post the initial placeholder message. Returns the message ts, or None on failure."""
        try:
            result = self._client.chat_postMessage(
                channel=self._channel,
                text=self._initial,
                thread_ts=self._thread_ts,
            )
            self.ts = result.get("ts")
            return self.ts
        except Exception as e:
            logger.error("Failed to post initial message: %s", e)
            return None

    def update(self, text: str, blocks: list = None) -> None:
        """Overwrite the message with new text. Throttled, never raises on API errors."""
        with self._lock:
            if not self.ts:
                raise RuntimeError("Cannot update before send()")

            # Throttle: skip if too soon since last update
            now = time.time()
            if now - self._last_update < self._MIN_UPDATE_INTERVAL:
                return
            self._last_update = now

        try:
            text = self._truncate(text)
            kwargs = {"channel": self._channel, "ts": self.ts, "text": text}
            if blocks:
                kwargs["blocks"] = blocks
            self._client.chat_update(**kwargs)
        except Exception as e:
            logger.warning("Progress update failed: %s", e)

    def finish(self, text: str, blocks: list = None) -> None:
        """Final overwrite with the response. Bypasses throttle. Never raises on API errors."""
        with self._lock:
            if not self.ts:
                raise RuntimeError("Cannot finish before send()")

        try:
            text = self._truncate(text)
            kwargs = {"channel": self._channel, "ts": self.ts, "text": text}
            if blocks:
                kwargs["blocks"] = blocks
            self._client.chat_update(**kwargs)
        except Exception as e:
            logger.warning("Final update failed: %s", e)

    def _truncate(self, text: str) -> str:
        """Truncate text to Slack's limit."""
        if len(text) > _MAX_TEXT_LENGTH:
            return text[:_MAX_TEXT_LENGTH] + "\n\n⚠️ Response truncated (too long for Slack)."
        return text
