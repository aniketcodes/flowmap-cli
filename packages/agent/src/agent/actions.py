"""Action/Approval Planner — LLM recommends, human approves via Slack buttons."""

import json
import logging
import re
import threading
import time
from pathlib import Path
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
    "create_bugfix_pr": {
        "id": "create_bugfix_pr",
        "label": "Create Bugfix PR",
        "description": "Generate a fix and create a pull request",
    },
    "create_pr": {
        "id": "create_pr",
        "label": "Create PR (Skip Tests)",
        "description": "Generate a fix and create a pull request without running tests",
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
- create_bugfix_pr: Generate a fix and create a pull request with tests (use when a specific bug is identified with a clear fix location)
- create_pr: Generate a fix and create a pull request without tests (use when you want a quick fix without TDD)

Respond with ONLY a JSON object:
{"actions": ["action_id_1", "action_id_2"]}

Rules:
- Pick the 1-2 most relevant actions based on BOTH the user's question and the diagnosis
- ALWAYS include "create_pr" if the diagnosis identifies a specific bug with a code location (preferred over create_bugfix_pr for speed)
- Include "show_history" if the user asks about changes, history, recent activity, or what changed
- Include "explain_code" if the diagnosis mentions specific files or code
- Include "create_thread" if the issue is complex and needs team discussion
- If unsure, default to ["create_pr", "explain_code"]"""


class ActionPlanner:
    """LLM-based action recommender."""

    def __init__(self, llm=None):
        self.llm = llm

    def recommend(self, diagnosis: str, tools_used: list[str] = None,
                  user_question: str = None) -> list[dict]:
        """Returns list of action dicts with id, label, description.
        
        Always includes create_pr and show_history.
        """
        action_ids = self._get_action_ids(diagnosis, tools_used, user_question)
        # Always include create_pr and show_history
        for mandatory in ["create_pr", "show_history"]:
            if mandatory not in action_ids:
                action_ids.append(mandatory)
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
        self._tdd_runner = None

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
        elif action_id == "create_bugfix_pr":
            result = self._create_bugfix_pr(message_ts, channel_id, diagnosis, thread_history)
        elif action_id == "create_pr":
            result = self._create_pr(message_ts, channel_id, diagnosis, thread_history)
        else:
            return f"Unknown action: {action_id}"
        
        # Mark as executed
        self._executed[exec_key] = datetime.now()
        return result

    def _create_bugfix_pr(self, message_ts: str, channel_id: str, diagnosis: str,
                          thread_history: list = None) -> str:
        """Create a bugfix PR using TDD approach."""
        from agent.bot import _md_to_slack

        # Find repo path from diagnosis
        repo_path = self._find_repo_from_diagnosis(diagnosis)
        if not repo_path:
            return "Could not determine repository from diagnosis"

        # Create BugfixPRCreator
        llm = getattr(self.agent, 'llm', None) if self.agent else None
        creator = BugfixPRCreator(flowmap=self.flowmap, llm=llm)

        # Run the full flow
        def progress(text):
            if self.client:
                try:
                    self.client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=message_ts,
                        text=f"⏳ {text}",
                    )
                except Exception:
                    pass

        result = creator.create_bugfix_pr(repo_path, diagnosis, thread_history=thread_history, on_progress=progress)

        if result["status"] == "success":
            return result["pr_url"]
        elif result["status"] == "error":
            return f"Error: {result.get('error', 'Unknown error')}"
        else:
            return f"Unexpected status: {result.get('status')}"

    def _create_pr(self, message_ts: str, channel_id: str, diagnosis: str,
                   thread_history: list = None) -> str:
        """Create a PR without TDD — just fix and push."""
        from agent.bot import _md_to_slack

        repo_path = self._find_repo_from_diagnosis(diagnosis)
        if not repo_path:
            return "Could not determine repository from diagnosis"

        llm = getattr(self.agent, 'llm', None) if self.agent else None
        creator = BugfixPRCreator(flowmap=self.flowmap, llm=llm)

        def progress(text):
            if self.client:
                try:
                    self.client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=message_ts,
                        text=f"⏳ {text}",
                    )
                except Exception:
                    pass

        result = creator.create_pr(repo_path, diagnosis, thread_history=thread_history, on_progress=progress)

        if result["status"] == "success":
            return result["pr_url"]
        elif result["status"] == "error":
            return f"Error: {result.get('error', 'Unknown error')}"
        else:
            return f"Unexpected status: {result.get('status')}"

    def _find_repo_from_diagnosis(self, diagnosis: str) -> str | None:
        """Ask the LLM which repo contains the bug that needs fixing."""
        import re

        repos = []
        if self.flowmap:
            try:
                repos = self.flowmap.call_tool("flowmap_repos", {})
            except Exception:
                pass

        if not repos:
            return None

        # Build a lookup: repo_name -> repo_path
        repo_map = {}
        repo_list = []
        for r in repos:
            name = r.get("name", "") if isinstance(r, dict) else str(r)
            path = r.get("path", "") if isinstance(r, dict) else ""
            repo_map[name] = path
            repo_list.append(name)

        # Ask the LLM which repo to fix
        llm = getattr(self.agent, 'llm', None) if self.agent else None
        if llm:
            prompt = (
                f"Given this diagnosis of a production issue, which repository contains the BUG "
                f"that needs to be fixed? Not where the error manifests, but where the ROOT CAUSE lives.\n\n"
                f"Available repositories: {', '.join(repo_list)}\n\n"
                f"Diagnosis:\n{diagnosis[:1000]}\n\n"
                f"Return ONLY the repository name (e.g., demo-order-service). Nothing else."
            )
            try:
                response = llm.chat(prompt)
                repo_name = response.content.strip().strip('`"\'')
                # Extract just the repo name if LLM returned extra text
                for name in repo_map:
                    if name in repo_name:
                        logger.info("find_repo llm_selected repo=%s", name)
                        return repo_map[name]
            except Exception as e:
                logger.warning("find_repo llm_failed error=%s", e)

        # Fallback: first repo mentioned
        for repo_name, repo_path in repo_map.items():
            if repo_name in diagnosis:
                return repo_path

        return None

    def _file_in_repo(self, repo_path: str, file_path: str) -> bool:
        """Check if a file exists in the repo (flexible matching)."""
        from pathlib import Path
        repo = Path(repo_path)
        # Try exact match
        if (repo / file_path).exists():
            return True
        # Try finding by filename only
        filename = Path(file_path).name
        matches = list(repo.rglob(filename))
        return len(matches) > 0

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


# ---------------------------------------------------------------------------
# Phase 5: Bugfix PR Creator — TDD-driven fix generation
# ---------------------------------------------------------------------------

class BugfixPRCreator:
    """Orchestrates the bugfix PR creation flow: read code → test → fix → PR."""

    def __init__(self, flowmap=None, llm=None):
        from agent.tdd_runner import TDDRunner
        from agent.git_ops import GitOperations
        self.flowmap = flowmap
        self.llm = llm
        self.tdd = TDDRunner()
        self.git = GitOperations()

    def create_bugfix_pr(self, repo_path: str, diagnosis: str,
                         thread_history: list = None, on_progress: callable = None) -> dict:
        """Full flow: read code, generate test, generate fix, push, create PR.

        Returns dict with:
            - status: "success" | "error" | "no_framework"
            - pr_url: PR URL (if success)
            - error: error message (if error)
        """
        from pathlib import Path

        repo = Path(repo_path)
        _progress = on_progress or (lambda x: None)

        # 1. Detect framework
        _progress("Detecting test framework...")
        framework = self.tdd.detect_framework(repo)
        if not framework:
            return {"status": "error", "error": "No test framework detected"}

        # 2. Install deps
        _progress(f"Installing dependencies ({framework})...")
        self.tdd.install_deps(repo, framework)

        # 3. Read source code via FlowMap
        _progress("Reading source code...")
        source_files = self._read_source_files(diagnosis, repo_path)
        if not source_files:
            return {"status": "error", "error": "Could not read source code from diagnosis"}

        # Combine all source for LLM context
        source_context = "\n\n".join(
            f"=== {path} ===\n{content[:8000]}"
            for path, content in source_files.items()
        )

        # Build export map so LLM knows which file exports what
        import re as _re
        export_map = {}
        for path, content in source_files.items():
            exports = _re.findall(r'^export\s+(?:function|const|class|{|default)\s+(\w+)', content, _re.MULTILINE)
            if exports:
                export_map[path] = exports
        export_hint = ""
        if export_map:
            export_hint = "\n\nEXPORTS BY FILE (use these for import paths):\n"
            for path, exports in export_map.items():
                export_hint += f"- {path}: {', '.join(exports)}\n"

        # Format thread history for context
        thread_context = ""
        if thread_history:
            thread_lines = []
            for msg in thread_history:
                role = msg.get("role", "user")
                text = msg.get("text", "")
                if text:
                    thread_lines.append(f"[{role}]: {text[:500]}")
            thread_context = "\n".join(thread_lines[-10:])  # Last 10 messages

        # Snapshot every file we touch so a failed attempt leaves the repo clean.
        originals: dict[str, "str | None"] = {}

        def _write(rel_path: str, content: str) -> None:
            target = repo / rel_path
            if rel_path not in originals:
                originals[rel_path] = target.read_text() if target.exists() else None
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

        def _restore() -> None:
            for rel_path, original in originals.items():
                target = repo / rel_path
                if original is None:
                    target.unlink(missing_ok=True)
                else:
                    target.write_text(original)

        # 4-9. Outer loop over test generations, inner loop over fix attempts.
        #      A test that fails to COMPILE/RUN (bad import, syntax error) is a
        #      broken test — not a valid RED — so it gets regenerated instead of
        #      burning fix attempts that can never make a non-compiling test pass.
        test_file = self._find_test_file(repo, framework)
        max_test_attempts = 3
        max_fix_attempts = 3
        test_error = None
        files = {}
        output = ""

        for test_attempt in range(1, max_test_attempts + 1):
            _progress(f"Generating failing test (attempt {test_attempt}/{max_test_attempts})...")
            test_content = self._generate_test(
                diagnosis, source_context, framework,
                test_file=test_file, export_hint=export_hint,
                thread_context=thread_context, previous_error=test_error,
            )
            _write(test_file, self._strip_code_fences(test_content))

            _progress("Running test (expecting RED)...")
            passed, output = self.tdd.run_test(repo, test_file, framework)

            # A compile/import failure means the TEST is broken — regenerate it.
            if self._test_broken(output):
                test_error = (
                    "The test file failed to compile/run. Fix its imports/syntax: import "
                    "ONLY from the source files shown, using their exact relative paths and "
                    f"exported names.\n\n{output[:400]}"
                )
                if test_attempt == max_test_attempts:
                    _restore()
                    return {"status": "error", "error": f"Could not generate a runnable test after {max_test_attempts} attempts.\n\n{output[:400]}"}
                _progress("Test file is broken, regenerating...")
                continue

            if passed:
                _restore()
                return {"status": "error", "error": "Test already passes — bug may not be reproducible"}

            # Test is a valid RED — now try to fix the code.
            fix_error = None
            fixed = False
            regen_test = False
            for fix_attempt in range(1, max_fix_attempts + 1):
                _progress(f"Generating fix (attempt {fix_attempt}/{max_fix_attempts})...")
                fix_content = self._generate_fix(
                    diagnosis, test_content, source_context, framework,
                    thread_context=thread_context, previous_error=fix_error,
                )
                files = self._parse_fix_output(fix_content)
                if not files:
                    fix_error = "No fix produced. Output each file as '=== path ===' followed by its full contents."
                    continue

                # Write the fix FIRST so validation and the GREEN run see the
                # actual fix in real project context — not the original file.
                for path, content in files.items():
                    _write(path, self._strip_code_fences(content))

                _progress("Validating syntax...")
                syntax_ok, syntax_err = self._validate_syntax(repo, files, framework)
                if not syntax_ok:
                    fix_error = f"Syntax error: {syntax_err}"
                    _progress("Syntax error, retrying...")
                    continue

                _progress("Running test (expecting GREEN)...")
                passed, output = self.tdd.run_test(repo, test_file, framework)
                if passed:
                    fixed = True
                    break

                # If the fix broke the test's ability to compile/run, the test's
                # imports no longer match the code — regenerate the test instead.
                if self._test_broken(output):
                    test_error = (
                        "After the fix, the test failed to compile/run — its imports no "
                        f"longer match the code. Rewrite the test to match.\n\n{output[:400]}"
                    )
                    regen_test = True
                    break

                fix_error = f"Test still failing:\n{output[:400]}"
                _progress("Test failed, retrying...")

            if fixed:
                break
            if regen_test and test_attempt < max_test_attempts:
                continue  # regenerate the test and try the fix loop again
            _restore()
            return {"status": "error", "error": f"Fix did not pass test after {max_fix_attempts} attempts.\n\n{fix_error or output[:400]}"}

        if not files:
            _restore()
            return {"status": "error", "error": "Failed to generate valid fix"}

        # 10. Create branch, commit, push, generate PR URL
        _progress("Pushing fix...")
        branch_name = self.git.generate_branch_name(diagnosis)

        if not self.git.create_branch(repo, branch_name):
            return {"status": "error", "error": "Failed to create branch"}

        if not self.git.apply_fix(repo, files):
            return {"status": "error", "error": "Failed to apply fix"}

        title = f"fix: {branch_name.split('/')[-1].replace('-', ' ')}"
        if not self.git.commit(repo, title):
            return {"status": "error", "error": "Failed to commit"}

        if not self.git.push(repo, branch_name):
            return {"status": "error", "error": f"Failed to push. Run: git push -u origin {branch_name}"}

        pr_url = self.git.get_pr_url(repo, branch_name)

        return {"status": "success", "pr_url": pr_url, "branch": branch_name}

    def create_pr(self, repo_path: str, diagnosis: str,
                  thread_history: list = None, on_progress: callable = None) -> dict:
        """Skip TDD — just read code, generate fix, push PR.

        Returns dict with:
            - status: "success" | "error"
            - pr_url: PR URL (if success)
            - error: error message (if error)
        """
        from pathlib import Path

        repo = Path(repo_path)
        _progress = on_progress or (lambda x: None)

        # 1. Read source code via FlowMap
        _progress("Reading source code...")
        source_files = self._read_source_files(diagnosis, repo_path)
        if not source_files:
            return {"status": "error", "error": "Could not read source code from diagnosis"}

        source_context = "\n\n".join(
            f"=== {path} ===\n{content[:8000]}"
            for path, content in source_files.items()
        )

        # 2. Format thread history
        thread_context = ""
        if thread_history:
            thread_lines = []
            for msg in thread_history:
                role = msg.get("role", "user")
                text = msg.get("text", "")
                if text:
                    thread_lines.append(f"[{role}]: {text[:500]}")
            thread_context = "\n".join(thread_lines[-10:])

        # 3. Generate fix via LLM
        _progress("Generating fix...")
        fix_content = self._generate_fix(
            diagnosis, "", source_context, "typescript",
            thread_context=thread_context,
        )
        files = self._parse_fix_output(fix_content)
        if not files:
            return {"status": "error", "error": "No fix produced by LLM"}

        # 4. Validate syntax
        _progress("Validating syntax...")
        syntax_ok, syntax_err = self._validate_syntax(repo, files, "typescript")
        if not syntax_ok:
            return {"status": "error", "error": f"Syntax error: {syntax_err}"}

        # 5. Create branch, commit, push
        _progress("Pushing fix...")
        branch_name = self.git.generate_branch_name(diagnosis)

        if not self.git.create_branch(repo, branch_name):
            return {"status": "error", "error": "Failed to create branch"}

        if not self.git.apply_fix(repo, files):
            return {"status": "error", "error": "Failed to apply fix"}

        title = f"fix: {branch_name.split('/')[-1].replace('-', ' ')}"
        if not self.git.commit(repo, title):
            return {"status": "error", "error": "Failed to commit"}

        if not self.git.push(repo, branch_name):
            return {"status": "error", "error": f"Failed to push. Run: git push -u origin {branch_name}"}

        pr_url = self.git.get_pr_url(repo, branch_name)

        return {"status": "success", "pr_url": pr_url, "branch": branch_name}

    def _read_source_files(self, diagnosis: str, repo_path: str) -> dict[str, str]:
        """Read source files mentioned in the diagnosis via FlowMap."""
        import re
        from pathlib import Path

        files = {}
        demo_repos = Path(repo_path).parent  # e.g., demo-repos/

        # Extract file paths from diagnosis (e.g., "src/order.ts:68" or "demo-order-service/src/order.ts:68")
        file_patterns = re.findall(r'`?([\w/-]+\.(?:ts|js|py|go|java|rs|tsx|jsx))`?', diagnosis)

        # Always search FlowMap for relevant source files in the selected repo
        if self.flowmap:
            try:
                repo_name = Path(repo_path).name
                results = self.flowmap.call_tool("flowmap_search", {
                    "query": diagnosis[:200],
                    "limit": 10,
                    "repo": repo_name,
                })
                for r in results:
                    if hasattr(r, 'file'):
                        file_path = r.file
                        if file_path.startswith(repo_path):
                            file_path = str(Path(file_path).relative_to(repo_path))
                        if 'REPO_GUIDE' not in file_path and 'README' not in file_path:
                            file_patterns.append(file_path)
                    elif isinstance(r, dict) and 'file' in r:
                        fp = r['file']
                        if 'REPO_GUIDE' not in fp and 'README' not in fp:
                            file_patterns.append(fp)
            except Exception as e:
                logger.warning("flowmap_repo_search_failed error=%s", e)

        # Also try searching by filename keywords from diagnosis
        if not file_patterns and self.flowmap:
            # Extract keywords like "order", "payment", "JSON.parse"
            keywords = re.findall(r'\b(order|payment|transaction|handler|service|json|parse)\b', diagnosis, re.IGNORECASE)
            if keywords:
                try:
                    results = self.flowmap.call_tool("flowmap_search", {
                        "query": " ".join(keywords[:5]),
                        "limit": 10,
                    })
                    for r in results:
                        if hasattr(r, 'file'):
                            file_path = r.file
                            if file_path.startswith(repo_path):
                                file_path = str(Path(file_path).relative_to(repo_path))
                            file_patterns.append(file_path)
                        elif isinstance(r, dict) and 'file' in r:
                            file_patterns.append(r['file'])
                except Exception:
                    pass

        # Read each file directly from disk
        repo = Path(repo_path)
        for file_path in set(file_patterns):
            # Skip REPO_GUIDE.md and other non-code files
            if 'REPO_GUIDE' in file_path or 'README' in file_path:
                continue

            # Try exact path in selected repo
            full_path = repo / file_path
            if full_path.exists():
                files[file_path] = full_path.read_text()
                continue

            # If path has repo prefix (e.g., "demo-ledger-service/pkg/handler.go"), strip it and try sibling repos
            parts = Path(file_path).parts
            if len(parts) > 1 and parts[0].startswith('demo-'):
                rel_path = str(Path(*parts[1:]))  # e.g., "pkg/handler.go"
                # Try in sibling repos
                for sibling in demo_repos.iterdir():
                    if sibling.is_dir() and sibling != repo:
                        sibling_path = sibling / rel_path
                        if sibling_path.exists():
                            files[str(sibling_path.relative_to(demo_repos))] = sibling_path.read_text()
                            break
                continue

            # Try finding by filename
            filename = Path(file_path).name
            matches = [f for f in repo.rglob(filename) if 'node_modules' not in f.parts and '.git' not in f.parts]
            if matches:
                rel = matches[0].relative_to(repo)
                files[str(rel)] = matches[0].read_text()

        return files

    def _find_test_file(self, repo: Path, framework: str) -> str:
        """Detect where tests live in this repo and return path for new test."""
        import re

        def _find_tests(pattern: str) -> list[Path]:
            """Find test files excluding node_modules and vendor dirs."""
            return [
                f for f in repo.rglob(pattern)
                if 'node_modules' not in f.parts
                and 'vendor' not in f.parts
                and '.git' not in f.parts
            ]

        if framework == "jest":
            test_files = _find_tests("*.test.ts") + _find_tests("*.test.js")
            if test_files:
                return str(test_files[0].parent / "bugfix.test.ts")
            return "src/__tests__/bugfix.test.ts"

        elif framework == "pytest":
            test_files = _find_tests("test_*.py") + _find_tests("*_test.py")
            if test_files:
                return str(test_files[0].parent / "test_bugfix.py")
            return "tests/test_bugfix.py"

        elif framework == "go":
            go_files = _find_tests("*.go")
            if go_files:
                for f in go_files:
                    if "_test.go" not in f.name:
                        content = f.read_text()
                        pkg_match = re.search(r'^package\s+(\w+)', content, re.MULTILINE)
                        if pkg_match:
                            return str(f.parent / f"bugfix_{pkg_match.group(1)}_test.go")
            return "bugfix_test.go"

        return "tests/test_bugfix.py"

    def _generate_test(self, diagnosis: str, source_code: str, framework: str, test_file: str = None, export_hint: str = None, thread_context: str = None, previous_error: str = None) -> str:
        """Use LLM to generate a failing test."""
        if not self.llm:
            return self._fallback_test(diagnosis, framework)

        location_hint = ""
        if test_file:
            location_hint = f"\nTest file location: {test_file}\nUse correct relative import paths based on this location.\n"

        error_feedback = ""
        if previous_error:
            error_feedback = (
                f"\n\nPREVIOUS ATTEMPT FAILED:\n{previous_error}\n"
                f"Fix this error in your next attempt.\n"
            )

        prompt = (
            f"You are a senior test engineer. Write a test that FAILS with the current code.\n\n"
            f"Framework: {framework}\n\n"
            f"Diagnosis:\n{diagnosis}\n\n"
            f"Source code (READ THIS CAREFULLY — the bug is in this code):\n{source_code}\n"
            + (export_hint or "")
            + (f"\nFull diagnostic context (tool calls, search results, reasoning):\n{thread_context}\n\n" if thread_context else "")
            + f"{location_hint}"
            f"{error_feedback}\n"
            f"Requirements:\n"
            f"1. The source code above is the ACTUAL code with the bug — study it carefully\n"
            f"2. Write a test that demonstrates the bug described in the diagnosis\n"
            f"3. Use the same test framework and conventions as the existing code\n"
            f"4. Mock external dependencies (HTTP calls, databases, etc.)\n"
            f"5. The test MUST fail with the current code (RED phase of TDD)\n"
            f"6. Output ONLY the test file content, no explanation\n"
            f"7. Do NOT hardcode specific IDs, values, or transaction numbers — use dynamic test data\n"
            f"8. Import from the CORRECT file — look at EXPORTS BY FILE above and import from the file that exports the symbol you need. Do NOT import from index unless the symbol is exported there.\n"
        )
        logger.info("_generate_test prompt_length=%d export_hint=%s", len(prompt), repr(export_hint[:200]) if export_hint else "None")
        try:
            response = self.llm.chat(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error("LLM test generation failed: %s", e)
            return self._fallback_test(diagnosis, framework)

    def _generate_fix(self, diagnosis: str, test_content: str,
                      source_code: str, framework: str, thread_context: str = None, previous_error: str = None) -> str:
        """Use LLM to generate the fix."""
        if not self.llm:
            return ""

        error_feedback = ""
        if previous_error:
            error_feedback = (
                f"\n\nPREVIOUS ATTEMPT FAILED WITH THIS ERROR:\n{previous_error}\n"
                f"Fix this error in your next attempt. Do NOT repeat the same mistake.\n"
            )

        prompt = (
            f"You are a senior engineer. Fix the bug in this code.\n\n"
            f"Diagnosis:\n{diagnosis}\n\n"
            f"Source code (READ THIS CAREFULLY — the bug is in this code):\n{source_code}\n\n"
            + (f"Full diagnostic context (tool calls, search results, reasoning):\n{thread_context}\n\n" if thread_context else "")
            + f"The test that must pass:\n{test_content}\n\n"
            f"Requirements:\n"
            f"1. The source code above is the ACTUAL code with the bug — study it carefully\n"
            f"2. Fix must make the test pass\n"
            f"3. Follow the project's existing code style\n"
            f"4. Don't add unrelated changes\n"
            f"5. Use RELATIVE paths only (e.g., src/order.ts)\n"
            f"6. For multi-file fixes, output each file with its path\n"
            f"7. Do NOT hardcode specific IDs, values, or transaction numbers — the fix must be GENERIC\n"
            f"8. Fix the ROOT CAUSE (e.g., precision loss in parsing), not the symptom\n"
            f"9. Do NOT create or modify files in node_modules, vendor, dist, build, or .git directories\n"
            f"10. Keep the public API stable — do NOT remove, rename, or change the signature of "
            f"exported functions/values the test imports, or the test will fail to compile\n"
            f"{error_feedback}\n"
            f"Output format:\n"
            f"=== src/file1.ts ===\n"
            f"{{fixed content}}\n\n"
            f"=== src/file2.ts ===\n"
            f"{{fixed content}}\n"
        )
        try:
            response = self.llm.chat(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error("LLM fix generation failed: %s", e)
            return ""

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences (```typescript ... ```) from LLM output."""
        import re
        # Remove opening fence with optional language tag
        text = re.sub(r'^```\w*\s*\n', '', text.strip())
        # Remove closing fence
        text = re.sub(r'\n```\s*$', '', text)
        return text.strip()

    @staticmethod
    def _test_broken(output: str) -> bool:
        """True when the test failed to COMPILE/RUN (an infrastructure problem
        — bad import, syntax error, empty collection) rather than running and
        failing an assertion (a legitimate RED/GREEN signal). A broken test can
        never be made to pass by fixing the code, so it must be regenerated.
        Framework-agnostic across jest/ts-jest, pytest, and go test.
        """
        if not output:
            return False
        # Any TypeScript compiler diagnostic (TS2305 wrong import, TS2307 missing
        # module, etc.) means the test/source did not type-check.
        if re.search(r"error TS\d+", output):
            return True
        markers = (
            "Test suite failed to run",   # jest / ts-jest compile failure
            "Cannot find module",
            "has no exported member",     # TS2305 — importing a symbol that doesn't exist
            "SyntaxError",
            "ModuleNotFoundError",        # pytest import failure
            "ImportError",
            "collected 0 items",          # pytest collected nothing
            "no tests found",             # jest matched nothing
            "[build failed]",             # go test compile failure
            "cannot find package",        # go
        )
        low = output.lower()
        return any(m.lower() in low for m in markers)

    def _validate_syntax(self, repo: Path, files: dict[str, str], framework: str) -> tuple[bool, str]:
        """Validate that the written fix compiles/parses. Returns (ok, error_message).

        Assumes the fix has already been written to disk. Checks are project-aware
        where imports matter (TypeScript via the repo's tsconfig, Go via the whole
        module) so a valid file with imports isn't rejected for missing context.
        Only errors that implicate a file we changed count as a failure — this
        ignores pre-existing/unrelated project errors. A missing toolchain or a
        timeout is treated as "skip" (ok), since the GREEN test run is the real gate.
        """
        import subprocess

        def _run(cmd: list, timeout: int):
            try:
                r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, timeout=timeout)
                return r.returncode, (r.stdout + r.stderr).strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return None, ""  # toolchain unavailable / too slow — don't block

        def _implicates(output: str, paths: list) -> bool:
            return any(p in output or Path(p).name in output for p in paths)

        changed = list(files.keys())

        if framework == "jest":
            ts_files = [p for p in changed if p.endswith((".ts", ".tsx"))]
            js_files = [p for p in changed if p.endswith((".js", ".jsx", ".mjs", ".cjs"))]
            if ts_files:
                if (repo / "tsconfig.json").exists():
                    # Type-check the project so imports resolve as they do at runtime.
                    cmd = ["npx", "tsc", "--noEmit", "--skipLibCheck", "-p", "tsconfig.json"]
                else:
                    cmd = ["npx", "tsc", "--noEmit", "--skipLibCheck", "--allowJs",
                           "--esModuleInterop", "--moduleResolution", "node",
                           "--target", "es2020", "--module", "commonjs"] + \
                          [str(repo / p) for p in ts_files]
                code, out = _run(cmd, 90)
                if code not in (None, 0) and _implicates(out, ts_files):
                    return False, f"TypeScript error:\n{out[:800]}"
            for p in js_files:
                code, out = _run(["node", "--check", str(repo / p)], 15)
                if code not in (None, 0):
                    return False, f"JavaScript syntax error in {p}:\n{out[:400]}"

        elif framework == "pytest":
            for p in changed:
                if p.endswith(".py"):
                    code, out = _run(["python", "-m", "py_compile", str(repo / p)], 15)
                    if code not in (None, 0):
                        return False, f"Python syntax error in {p}:\n{out[:400]}"

        elif framework == "go":
            # Build the whole module so package context and imports are available.
            go_files = [p for p in changed if p.endswith(".go")]
            code, out = _run(["go", "build", "./..."], 90)
            if code not in (None, 0) and (not go_files or _implicates(out, go_files)):
                return False, f"Go build error:\n{out[:800]}"

        return True, ""

    def _parse_fix_output(self, fix_content: str) -> dict[str, str]:
        """Parse LLM output into {path: content} dict."""
        import re
        files = {}
        parts = re.split(r"===\s*(.+?)\s*===", fix_content)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                path = parts[i].strip()
                content = parts[i + 1].strip()
                # Validate path is relative and safe
                if not path.startswith("/") and ".." not in path:
                    files[path] = content

        return files

    def _compute_diff(self, repo: Path, files: dict[str, str]) -> str:
        """Compute unified diff between original and fixed files."""
        import difflib
        diffs = []
        for rel_path, new_content in files.items():
            original_path = repo / rel_path
            if original_path.exists():
                old_content = original_path.read_text()
            else:
                old_content = ""
            diff = difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
                n=3,
            )
            diffs.append("".join(diff))
        return "\n".join(diffs)

    def _format_pr_body(self, diagnosis: str, files: dict[str, str],
                        test_content: str, test_passed: bool, test_output: str) -> str:
        """Format detailed PR description."""
        files_list = "\n".join(f"- `{f}`" for f in files.keys())
        test_status = "Passing" if test_passed else "Needs review"

        return (
            "## Bug Fix\n\n"
            f"**Root Cause:**\n{diagnosis[:500]}\n\n"
            "## Changes\n\n"
            f"{files_list}\n\n"
            "## Test\n\n"
            f"**Status:** {test_status}\n\n"
            f"```\n{test_output[:500]}\n```\n\n"
            "## Generated by FlowMap\n\n"
            "This PR was auto-generated by FlowMap's Bugfix PR Creator.\n"
            "Please review the changes carefully before merging."
        )

    def _fallback_test(self, diagnosis: str, framework: str) -> str:
        """Fallback test when LLM is unavailable."""
        if framework == "jest":
            return (
                "describe('bugfix', () => {\n"
                "  it('should handle the reported issue', () => {\n"
                f"    // TODO: {diagnosis[:100]}\n"
                "    expect(true).toBe(false);\n"
                "  });\n"
                "});\n"
            )
        elif framework == "pytest":
            return (
                "def test_bugfix():\n"
                f"    # TODO: {diagnosis[:100]}\n"
                "    assert False, 'Bug not yet fixed'\n"
            )
        elif framework == "go":
            return (
                "package main\n\n"
                "import \"testing\"\n\n"
                f"func TestBugfix(t *testing.T) {{\n"
                f"    // TODO: {diagnosis[:100]}\n"
                "    t.Fatal(\"Bug not yet fixed\")\n"
                "}\n"
            )
        return "def test_bugfix():\n    assert False\n"

