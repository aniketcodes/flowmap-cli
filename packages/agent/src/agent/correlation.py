"""Temporal Correlation - Correlate code changes with metric spikes."""

from datetime import datetime


def correlate(commit_ts: datetime, metric_ts: datetime) -> float:
    """Calculate confidence score based on temporal proximity."""
    # Calculate time difference in minutes
    diff_minutes = abs((metric_ts - commit_ts).total_seconds() / 60)

    # 4-hour window (240 minutes)
    window_minutes = 240

    # Outside window → 0.0
    if diff_minutes > window_minutes:
        return 0.0

    # Within window → linear decay from 1.0 to 0.0
    confidence = 1.0 - (diff_minutes / window_minutes)
    return max(0.0, min(1.0, confidence))
