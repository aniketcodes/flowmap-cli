"""Security utilities - Signature verification, rate limiting, dedup."""

import time
import hashlib
import hmac
import logging
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 30  # requests per window per IP
EVENT_TTL = 600  # 10 minutes — Slack can replay events slowly

# Rate limiting state
_rate_limits: dict[str, list[float]] = defaultdict(list)


def verify_slack_signature(signing_secret: str, timestamp: str, signature: str, body: bytes) -> bool:
    """Verify Slack request signature to prevent spoofed requests."""
    if not signing_secret:
        logger.warning("No SLACK_SIGNING_SECRET configured — skipping verification")
        return True

    try:
        if abs(time.time() - float(timestamp)) > 300:
            logger.warning("Slack request timestamp too old: %s", timestamp)
            return False
    except (ValueError, TypeError):
        logger.warning("Invalid Slack request timestamp: %s", timestamp)
        return False

    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    computed = hmac.new(
        signing_secret.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()
    expected = f"v0={computed}"
    return hmac.compare_digest(expected, signature)


def check_rate_limit(ip: str) -> bool:
    """Return True if request is allowed, False if rate limited.

    Cleans up stale entries for the requesting IP on each call.
    Other IPs are cleaned up lazily when they make a request.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    _rate_limits[ip] = [t for t in _rate_limits[ip] if t > window_start]
    if len(_rate_limits[ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_limits[ip].append(now)
    return True


def cleanup_rate_limits() -> int:
    """Remove stale entries from all IPs. Returns number of IPs removed."""
    cutoff = time.time() - RATE_LIMIT_WINDOW
    removed = 0
    empty_ips = []
    for ip, timestamps in _rate_limits.items():
        _rate_limits[ip] = [t for t in timestamps if t > cutoff]
        if not _rate_limits[ip]:
            empty_ips.append(ip)
            removed += 1
    for ip in empty_ips:
        del _rate_limits[ip]
    return removed


class TTLCache:
    """Thread-safe TTL cache for deduplicating Slack events."""

    def __init__(self, ttl: int = EVENT_TTL):
        self._ttl = ttl
        self._items: dict[str, float] = {}
        self._lock = threading.Lock()

    def contains_and_add(self, key: str) -> bool:
        """Atomic: return True if key already exists, then add it.

        Thread-safe: only one thread per key gets False.
        """
        with self._lock:
            self._evict()
            if key in self._items:
                return True
            self._items[key] = time.time()
            return False

    def contains(self, key: str) -> bool:
        with self._lock:
            self._evict()
            return key in self._items

    def add(self, key: str) -> None:
        with self._lock:
            self._items[key] = time.time()

    def __len__(self) -> int:
        with self._lock:
            self._evict()
            return len(self._items)

    def _evict(self) -> None:
        cutoff = time.time() - self._ttl
        self._items = {k: v for k, v in self._items.items() if v > cutoff}
