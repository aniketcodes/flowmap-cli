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

_PLANNER_SYSTEM_PROMPT = """You are an operations assistant. Given a user's question and a diagnosis of a production issue,
recommend up to 2 actions the user can take.

Available actions:
- create_thread: Open a thread for team collaboration
- explain_code: Get a detailed explanation of the code path
- notify_team: Post a summary to the team channel
- show_history: Display recent commits affecting this code

Respond with ONLY a JSON object:
{"actions": ["action_id_1", "action_id_2"]}

Rules:
- Pick the 1-2 most relevant actions based on BOTH the user's question and the diagnosis
- Include "show_history" if the user asks about changes, history, recent activity, or what changed
- Include "show_history" if the diagnosis references specific commits or commit messages
- Include "explain_code" if the diagnosis mentions specific files or code
- Include "create_thread" if the issue is complex and needs team discussion
- If unsure, default to ["create_thread", "explain_code"]"""


class ActionPlanner:
    """LLM-based action recommender."""

    def __init__(self, llm=None):
        self.llm = llm

    def recommend(self, diagnosis: str, tools_used: list[str] = None,
                  user_question: str = None) -> list[dict]:
        """Returns list of action dicts with id, label, description.
        
        show_history is always included — seeing recent changes is universally useful.
        """
        action_ids = self._get_action_ids(diagnosis, tools_used, user_question)
        # Always include show_history
        if "show_history" not in action_ids:
            action_ids.append("show_history")
        # Max 3 actions
        action_ids = action_ids[:3]
        return [_ALL_ACTIONS[aid] for aid in action_ids if aid in _ALL_ACTIONS]

    def _get_action_ids(self, diagnosis: str, tools_used: list[str] = None,
                        user_question: str = None) -> list[str]:
        if not self.llm:
            logger.info("No LLM, using default actions")
            return list(_DEFAULT_ACTIONS)

        try:
            user_content = f"User's question:\n{user_question or '(not provided)'}\n\nDiagnosis:\n{diagnosis}"
            messages = [
                {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompt = "\n".join(m["content"] for m in messages)
            logger.info("ActionPlanner calling LLM...")
            response = self.llm.chat(prompt)
            logger.info("ActionPlanner LLM response: %.200s", response.content)
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
    """Build Slack Block Kit blocks with action buttons.

    Does NOT include diagnosis text — that's already in the streaming message.
    """
    blocks = []

    if actions:
        blocks.append({"type": "divider"})

        elements = []
        for action in actions[:4]:  # Max 4 buttons
            elements.append({
                "type": "button",
                "text": {"type": "plain_text", "text": action["label"], "emoji": True},
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

    def __init__(self, client=None, agent=None, flowmap=None):
        self.client = client
        self.agent = agent
        self.flowmap = flowmap
        self._executed = {}  # { (action_id, message_ts): datetime }
        self._ttl_hours = 24

    def execute(self, action_id: str, message_ts: str, channel_id: str, diagnosis: str,
                thread_history: list = None) -> str:
        # Idempotency check
        from datetime import datetime, timedelta
        exec_key = (action_id, message_ts)
        if exec_key in self._executed:
            exec_time = self._executed[exec_key]
            if datetime.now() - exec_time < timedelta(hours=self._ttl_hours):
                return "Already executed"
        
        if action_id == "create_thread":
            result = self._create_thread(message_ts, channel_id, diagnosis)
        elif action_id == "explain_code":
            result = self._explain_code(message_ts, channel_id, diagnosis, thread_history)
        elif action_id == "notify_team":
            result = self._notify_team(channel_id, diagnosis)
        elif action_id == "show_history":
            result = self._show_history(message_ts, channel_id, diagnosis, thread_history)
        else:
            return f"Unknown action: {action_id}"
        
        # Mark as executed
        self._executed[exec_key] = datetime.now()
        return result

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

    def _explain_code(self, message_ts: str, channel_id: str, diagnosis: str,
                      thread_history: list = None) -> str:
        """Explain code using agent with focused prompt."""
        if not self.agent:
            return "Agent not available"
        
        try:
            prompt = f"Explain the code related to this diagnosis in detail:\n{diagnosis}"
            if thread_history:
                context = "\n".join([f"- {msg.get('text', '')}" for msg in thread_history[:5]])
                prompt += f"\n\nConversation context:\n{context}"
            prompt += "\n\nRead the relevant files and explain the code flow, key functions, and how they work together."
            explanation = self.agent.diagnose(prompt, max_steps=8)
        except Exception as e:
            return f"Failed to explain code: {e}"
        
        from agent.bot import _md_to_slack
        self.client.chat_postMessage(
            channel=channel_id,
            thread_ts=message_ts,
            text=_md_to_slack(f"*Code Explanation:*\n\n{explanation}"),
        )
        return "Explanation posted"

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

    def _show_history(self, message_ts: str, channel_id: str, diagnosis: str,
                      thread_history: list = None) -> str:
        """Show recent commits using FlowMap, with LLM-extracted query from thread context.
        
        Searches across ALL repos (no string-matching for repo name).
        Results grouped by repo — handles cross-repo changes naturally.
        """
        if not self.flowmap:
            return "FlowMap not available"

        # Get available repos for LLM context
        try:
            repos_data = self.flowmap.call_tool("flowmap_repos", {})
            known_repos = [
                r.get("name", str(r)) if isinstance(r, dict) else str(r)
                for r in (repos_data or [])
            ]
        except Exception:
            known_repos = []

        # Build context from thread history + diagnosis
        context_parts = []
        if thread_history:
            context_parts.extend([f"- {msg.get('text', '')}" for msg in thread_history[:10]])
        if diagnosis:
            context_parts.append(f"- Diagnosis: {diagnosis[:500]}")
        context = "\n".join(context_parts)

        # Use LLM to extract a concise search query from conversation
        query = None
        llm = getattr(self.agent, 'llm', None) if self.agent else None
        if llm and context:
            prompt = (
                "Extract a concise search query (2-5 words) for finding relevant git commits "
                "based on this conversation. The query should match commit messages.\n\n"
                f"Known repos: {', '.join(known_repos)}\n\n"
                f"Conversation:\n{context}\n\n"
                'Return JSON: {"query": "concise search terms"}\n'
                "Example: {\"query\": \"redis timeout config\"}"
            )
            try:
                response = llm.chat(prompt)
                import json
                raw = response.content.strip()
                # Strip markdown code fences if present (```json ... ```)
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    raw = raw.strip()
                # Try parsing as JSON
                try:
                    data = json.loads(raw)
                    query = data.get("query")
                except json.JSONDecodeError:
                    # LLM might return plain text — use it directly if short
                    text = raw.strip('"').strip("'")
                    if text and len(text) < 200 and not text.startswith("```"):
                        query = text
            except Exception as e:
                logger.warning("show_history llm_call_failed error=%s", e)
                query = None

        # Fallback: extract keywords from diagnosis (not full sentence)
        if not query:
            if diagnosis:
                # Take first few meaningful words
                words = diagnosis.split()[:5]
                query = " ".join(words)
            else:
                query = "recent changes"

        logger.info("show_history query=%s", query[:100])

        # Call flowmap_history (no repo filter → searches ALL repos)
        try:
            result = self.flowmap.call_tool("flowmap_history", {"query": query})
        except Exception as e:
            return f"Failed to fetch history: {e}"

        # Group commits by repo (deduplicated by SHA)
        from agent.server import CommitInfo
        if isinstance(result, list) and result and isinstance(result[0], CommitInfo):
            sections = self._format_commits_by_repo(result)
        elif isinstance(result, str):
            sections = result
        else:
            sections = self._format_commits_by_repo(result)

        # Post to thread
        from agent.bot import _md_to_slack
        self.client.chat_postMessage(
            channel=channel_id,
            thread_ts=message_ts,
            text=_md_to_slack(f"*Recent Changes:*\n\n{sections}"),
        )
        return "History posted"

    @staticmethod
    def _format_commits_by_repo(commits) -> str:
        """Format commits grouped by repo name, deduplicated by SHA."""
        by_repo = {}
        seen_shas = set()
        for c in commits:
            sha = getattr(c, 'sha', '???')
            if sha in seen_shas:
                continue
            seen_shas.add(sha)
            repo = getattr(c, 'repo', 'unknown')
            by_repo.setdefault(repo, []).append(c)

        lines = []
        for repo, repo_commits in by_repo.items():
            lines.append(f"*{repo}*")
            for c in repo_commits:
                sha = getattr(c, 'sha', '???')[:7]
                msg = getattr(c, 'message', '').split('\n')[0][:80]
                lines.append(f"  • `{sha}` {msg}")
            lines.append("")
        return "\n".join(lines).strip()
