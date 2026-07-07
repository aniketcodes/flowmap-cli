"""Action/Approval Planner — LLM recommends, human approves via Slack buttons."""

import json
import logging
import re
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 1: ActionRegistry — stores actions + diagnosis by message_ts, TTL-based
# ---------------------------------------------------------------------------

class ActionRegistry:
    """Thread-safe store for actions by message_ts with TTL eviction."""

    def __init__(self, ttl: int = 3600):
        self._store: dict[str, dict] = {}
        self._timestamps: dict[str, float] = {}
        self._ttl = ttl
        self._lock = threading.Lock()

    def store(self, message_ts: str, actions: list[dict], diagnosis: str = ""):
        with self._lock:
            self._store[message_ts] = {"actions": actions, "diagnosis": diagnosis}
            self._timestamps[message_ts] = time.time()

    def get(self, message_ts: str) -> Optional[dict]:
        with self._lock:
            ts = self._timestamps.get(message_ts)
            if ts is None or time.time() - ts > self._ttl:
                self._store.pop(message_ts, None)
                self._timestamps.pop(message_ts, None)
                return None
            return self._store.get(message_ts)

    def get_actions(self, message_ts: str) -> Optional[list[dict]]:
        entry = self.get(message_ts)
        return entry["actions"] if entry else None

    def get_diagnosis(self, message_ts: str) -> str:
        entry = self.get(message_ts)
        return entry["diagnosis"] if entry else "Unknown issue"

    def __len__(self) -> int:
        with self._lock:
            self._evict()
            return len(self._store)

    def _evict(self):
        now = time.time()
        expired = [k for k, ts in self._timestamps.items() if now - ts > self._ttl]
        for k in expired:
            self._store.pop(k, None)
            self._timestamps.pop(k, None)


# ---------------------------------------------------------------------------
# Phase 2: ActionPlanner — LLM-based action recommendations
# ---------------------------------------------------------------------------

_ALL_ACTIONS = {
    "create_thread": {
        "id": "create_thread",
        "label": "Create Investigation Thread",
        "description": "Open a thread for team collaboration",
    },
    "explain_code": {
        "id": "explain_code",
        "label": "Explain Code Flow",
        "description": "Get a detailed explanation of the code path",
    },
    "notify_team": {
        "id": "notify_team",
        "label": "Notify Team",
        "description": "Post a summary to the team channel",
    },
    "show_history": {
        "id": "show_history",
        "label": "Show Recent Changes",
        "description": "Display recent commits affecting this code",
    },
}

_DEFAULT_ACTIONS = ["create_thread", "explain_code"]

_PLANNER_SYSTEM_PROMPT = """You are an operations assistant. Given a diagnosis of a production issue,
recommend up to 2 actions the user can take.

Available actions:
- create_thread: Open a thread for team collaboration
- explain_code: Get a detailed explanation of the code path
- notify_team: Post a summary to the team channel
- show_history: Display recent commits affecting this code

Respond with ONLY a JSON object:
{"actions": ["action_id_1", "action_id_2"]}

Rules:
- Pick the 1-2 most relevant actions
- Always include "explain_code" if the diagnosis mentions specific files or code
- Always include "create_thread" if the issue is complex
- If unsure, default to ["create_thread", "explain_code"]"""


class ActionPlanner:
    """LLM-based action recommender."""

    def __init__(self, llm=None):
        self.llm = llm

    def recommend(self, diagnosis: str, tools_used: list[str] = None) -> list[dict]:
        """Returns list of action dicts with id, label, description."""
        action_ids = self._get_action_ids(diagnosis, tools_used)
        return [_ALL_ACTIONS[aid] for aid in action_ids if aid in _ALL_ACTIONS]

    def _get_action_ids(self, diagnosis: str, tools_used: list[str] = None) -> list[str]:
        if not self.llm:
            return list(_DEFAULT_ACTIONS)

        try:
            messages = [
                {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Diagnosis:\n{diagnosis}"},
            ]
            response = self.llm.chat("\n".join(m["content"] for m in messages))
            return self._parse_response(response.content)
        except Exception as e:
            logger.warning("ActionPlanner LLM error: %s, using defaults", e)
            return list(_DEFAULT_ACTIONS)

    def _parse_response(self, text: str) -> list[str]:
        """Safely extract action IDs from LLM response."""
        # Try to find JSON in the response
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                actions = data.get("actions", [])
                if isinstance(actions, list) and all(isinstance(a, str) for a in actions):
                    return actions[:2]  # Max 2 actions
            except json.JSONDecodeError:
                pass

        # Fallback: look for known action IDs in text
        found = [aid for aid in _ALL_ACTIONS if aid in text.lower()]
        return found[:2] if found else list(_DEFAULT_ACTIONS)


# ---------------------------------------------------------------------------
# Phase 3: Block Kit builder — interactive message with action buttons
# ---------------------------------------------------------------------------

def build_action_blocks(
    diagnosis: str, actions: list[dict], message_ts: str
) -> list[dict]:
    """Build Slack Block Kit blocks with action buttons."""
    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": diagnosis[:3000]},
        },
        {"type": "divider"},
    ]

    if actions:
        elements = []
        for action in actions[:4]:  # Max 4 buttons
            elements.append({
                "type": "button",
                "text": {"type": "plain_text", "text": action["label"]},
                "action_id": f"{action['id']}|{message_ts}",
                "value": action["id"],
            })

        blocks.append({
            "type": "actions",
            "elements": elements,
        })

    return blocks


# ---------------------------------------------------------------------------
# Phase 4: ActionExecutor — runs approved actions
# ---------------------------------------------------------------------------

class ActionExecutor:
    """Executes approved actions via Slack API."""

    def __init__(self, client=None):
        self.client = client

    def execute(self, action_id: str, message_ts: str, channel_id: str, diagnosis: str) -> str:
        if action_id == "create_thread":
            return self._create_thread(message_ts, channel_id, diagnosis)
        elif action_id == "explain_code":
            return self._explain_code(diagnosis)
        elif action_id == "notify_team":
            return self._notify_team(channel_id, diagnosis)
        elif action_id == "show_history":
            return self._show_history(diagnosis)
        return f"Unknown action: {action_id}"

    def _create_thread(self, message_ts: str, channel_id: str, diagnosis: str) -> str:
        if not self.client:
            return "Slack client not available"
        try:
            self.client.chat_postMessage(
                channel=channel_id,
                thread_ts=message_ts,
                text=f"Investigation started for: {diagnosis[:200]}",
            )
            return "Thread created"
        except Exception as e:
            return f"Failed to create thread: {e}"

    def _explain_code(self, diagnosis: str) -> str:
        return f"Code explanation will be provided for the identified issue."

    def _notify_team(self, channel_id: str, diagnosis: str) -> str:
        if not self.client:
            return "Slack client not available"
        try:
            self.client.chat_postMessage(
                channel=channel_id,
                text=f":rotating_light: *Production Issue Detected*\n{diagnosis[:500]}",
            )
            return "Team notified"
        except Exception as e:
            return f"Failed to notify team: {e}"

    def _show_history(self, diagnosis: str) -> str:
        return "Recent commit history will be displayed."
