"""Tests for Bugfix PR Creator — written FIRST (TDD)."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import subprocess


# ── TDDRunner Tests ─────────────────────────────────────────────────────────

class TestTDDRunnerDetectFramework:
    """Test framework detection from repo structure."""

    def test_detect_jest_from_config(self, tmp_path):
        """Detects Jest when jest.config.js exists."""
        from agent.tdd_runner import TDDRunner
        (tmp_path / "jest.config.js").write_text("module.exports = {}")
        (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
        runner = TDDRunner()
        assert runner.detect_framework(tmp_path) == "jest"

    def test_detect_pytest_from_ini(self, tmp_path):
        """Detects pytest when pytest.ini exists."""
        from agent.tdd_runner import TDDRunner
        (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests")
        runner = TDDRunner()
        assert runner.detect_framework(tmp_path) == "pytest"

    def test_detect_pytest_from_config(self, tmp_path):
        """Detects pytest when pyproject.toml has [tool.pytest]."""
        from agent.tdd_runner import TDDRunner
        (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']")
        runner = TDDRunner()
        assert runner.detect_framework(tmp_path) == "pytest"

    def test_detect_go_from_mod(self, tmp_path):
        """Detects Go test when go.mod exists."""
        from agent.tdd_runner import TDDRunner
        (tmp_path / "go.mod").write_text("module example.com/foo")
        runner = TDDRunner()
        assert runner.detect_framework(tmp_path) == "go"

    def test_detect_jest_from_test_files(self, tmp_path):
        """Detects Jest from *.test.ts files when no config exists."""
        from agent.tdd_runner import TDDRunner
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "order.test.ts").write_text("describe('order', () => {})")
        (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
        runner = TDDRunner()
        assert runner.detect_framework(tmp_path) == "jest"

    def test_detect_none_when_no_tests(self, tmp_path):
        """Returns None when no test framework detected."""
        from agent.tdd_runner import TDDRunner
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        runner = TDDRunner()
        assert runner.detect_framework(tmp_path) is None

    def test_detect_from_subdir(self, tmp_path):
        """Detects framework in subdirectory (monorepo)."""
        from agent.tdd_runner import TDDRunner
        (tmp_path / "services").mkdir()
        (tmp_path / "services" / "order").mkdir()
        (tmp_path / "services" / "order" / "go.mod").write_text("module order")
        runner = TDDRunner()
        assert runner.detect_framework(tmp_path) == "go"


class TestTDDRunnerInstallDeps:
    """Test dependency installation."""

    @patch("subprocess.run")
    def test_install_npm(self, mock_run, tmp_path):
        """Runs npm install for Jest projects."""
        from agent.tdd_runner import TDDRunner
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        runner = TDDRunner()
        result = runner.install_deps(tmp_path, "jest")
        assert result is True
        mock_run.assert_called_once()
        assert "npm" in mock_run.call_args.args[0] or "install" in str(mock_run.call_args)

    @patch("subprocess.run")
    def test_install_pip(self, mock_run, tmp_path):
        """Runs pip install for pytest projects."""
        from agent.tdd_runner import TDDRunner
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        runner = TDDRunner()
        result = runner.install_deps(tmp_path, "pytest")
        assert result is True

    @patch("subprocess.run")
    def test_install_go_mod(self, mock_run, tmp_path):
        """Runs go mod download for Go projects."""
        from agent.tdd_runner import TDDRunner
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        runner = TDDRunner()
        result = runner.install_deps(tmp_path, "go")
        assert result is True

    @patch("subprocess.run")
    def test_install_failure_returns_false(self, mock_run, tmp_path):
        """Returns False when install fails."""
        from agent.tdd_runner import TDDRunner
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="error")
        runner = TDDRunner()
        result = runner.install_deps(tmp_path, "jest")
        assert result is False


class TestTDDRunnerRunTest:
    """Test running tests and detecting RED/GREEN."""

    @patch("subprocess.run")
    def test_run_test_red(self, mock_run, tmp_path):
        """Detects test failure (RED phase)."""
        from agent.tdd_runner import TDDRunner
        mock_run.return_value = Mock(
            returncode=1,
            stdout="FAIL src/order.test.ts",
            stderr=""
        )
        runner = TDDRunner()
        passed, output = runner.run_test(tmp_path, "src/order.test.ts", "jest")
        assert passed is False
        assert "FAIL" in output

    @patch("subprocess.run")
    def test_run_test_green(self, mock_run, tmp_path):
        """Detects test pass (GREEN phase)."""
        from agent.tdd_runner import TDDRunner
        mock_run.return_value = Mock(
            returncode=0,
            stdout="PASS src/order.test.ts",
            stderr=""
        )
        runner = TDDRunner()
        passed, output = runner.run_test(tmp_path, "src/order.test.ts", "jest")
        assert passed is True
        assert "PASS" in output

    @patch("subprocess.run")
    def test_run_test_timeout(self, mock_run, tmp_path):
        """Handles test timeout gracefully."""
        from agent.tdd_runner import TDDRunner
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="jest", timeout=30)
        runner = TDDRunner()
        passed, output = runner.run_test(tmp_path, "test.py", "pytest")
        assert passed is False
        assert "timeout" in output.lower() or "timed out" in output.lower()


# ── GitOperations Tests ─────────────────────────────────────────────────────

class TestGitOperations:
    """Test git operations with flowmap-agent identity."""

    @patch("subprocess.run")
    def test_create_branch(self, mock_run, tmp_path):
        """Creates a new branch."""
        from agent.git_ops import GitOperations
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        ops = GitOperations()
        result = ops.create_branch(tmp_path, "flowmap/bugfix/snowflake-id-abc123")
        assert result is True
        # Verify git checkout -b was called
        call_args = mock_run.call_args
        assert "checkout" in str(call_args) or "branch" in str(call_args)

    @patch("subprocess.run")
    def test_commit_uses_flowmap_identity(self, mock_run, tmp_path):
        """Commit uses flowmap-agent identity via env vars."""
        from agent.git_ops import GitOperations
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        ops = GitOperations()
        ops.commit(tmp_path, "fix: handle Snowflake IDs")
        # Verify env vars were set
        call_kwargs = mock_run.call_args.kwargs
        assert "env" in call_kwargs
        env = call_kwargs["env"]
        assert env.get("GIT_AUTHOR_NAME") == "flowmap-agent"
        assert env.get("GIT_COMMITTER_NAME") == "flowmap-agent"

    @patch("subprocess.run")
    def test_push(self, mock_run, tmp_path):
        """Pushes branch to origin."""
        from agent.git_ops import GitOperations
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        ops = GitOperations()
        result = ops.push(tmp_path, "flowmap/bugfix/snowflake-id-abc123")
        assert result is True

    @patch("subprocess.run")
    def test_get_pr_url(self, mock_run, tmp_path):
        """Generates GitHub PR URL from remote."""
        from agent.git_ops import GitOperations
        mock_run.return_value = Mock(
            returncode=0,
            stdout="git@github.com:aniketcodes/demo-order-service.git",
            stderr=""
        )
        ops = GitOperations()
        url = ops.get_pr_url(tmp_path, "flowmap/bugfix/test-abc123")
        assert "github.com/aniketcodes/demo-order-service" in url
        assert "flowmap/bugfix/test-abc123" in url

    @patch("subprocess.run")
    def test_push_failure_returns_false(self, mock_run, tmp_path):
        """Returns False when push fails."""
        from agent.git_ops import GitOperations
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="rejected")
        ops = GitOperations()
        result = ops.push(tmp_path, "flowmap/bugfix/test")
        assert result is False


# ── Branch Naming Tests ─────────────────────────────────────────────────────

class TestBranchNaming:
    """Test branch name generation."""

    def test_branch_name_format(self):
        """Branch name follows convention: flowmap/bugfix/<slug>-<hash>."""
        from agent.git_ops import GitOperations
        ops = GitOperations()
        name = ops.generate_branch_name("JavaScript JSON.parse loses precision on large transaction IDs")
        assert name.startswith("flowmap/bugfix/")
        assert len(name) < 80  # Reasonable length

    def test_branch_name_deterministic(self):
        """Same input produces same branch name."""
        from agent.git_ops import GitOperations
        ops = GitOperations()
        name1 = ops.generate_branch_name("Snowflake ID precision loss")
        name2 = ops.generate_branch_name("Snowflake ID precision loss")
        assert name1 == name2

    def test_branch_name_unique_per_input(self):
        """Different inputs produce different branch names."""
        from agent.git_ops import GitOperations
        ops = GitOperations()
        name1 = ops.generate_branch_name("Snowflake ID bug")
        name2 = ops.generate_branch_name("Redis timeout bug")
        assert name1 != name2


# ── ActionExecutor._create_bugfix_pr Tests ───────────────────────────────────

class TestCreateBugfixPR:
    """Test the full bugfix PR creation flow."""

    @patch("agent.actions.BugfixPRCreator")
    def test_full_flow_red_then_green(self, MockCreator):
        """Full flow: test RED -> fix -> test GREEN -> show diff."""
        from agent.actions import ActionExecutor

        # Mock BugfixPRCreator
        creator = MockCreator.return_value
        creator.create_bugfix_pr.return_value = {
            "status": "success",
            "pr_url": "https://github.com/test/repo/pull/42",
        }

        executor = ActionExecutor(flowmap=Mock())
        # Mock repo detection
        executor._find_repo_from_diagnosis = Mock(return_value="/tmp/test-repo")
        result = executor._create_bugfix_pr(
            message_ts="123.456",
            channel_id="C123",
            diagnosis="txn_id 334570640465989632 loses precision in JSON.parse",
            thread_history=None,
        )

        assert "github.com" in result or "pr" in result.lower()

    @patch("agent.actions.BugfixPRCreator")
    def test_no_test_framework_fallback(self, MockCreator):
        """Falls back when no test framework detected."""
        from agent.actions import ActionExecutor

        creator = MockCreator.return_value
        creator.create_bugfix_pr.return_value = {
            "status": "error",
            "error": "No test framework detected",
        }

        executor = ActionExecutor(flowmap=Mock())
        executor._find_repo_from_diagnosis = Mock(return_value="/tmp/test-repo")
        result = executor._create_bugfix_pr(
            message_ts="123.456",
            channel_id="C123",
            diagnosis="some bug",
            thread_history=None,
        )

        assert "no test framework" in result.lower() or "error" in result.lower()

    @patch("agent.actions.BugfixPRCreator")
    def test_success_returns_pr_url(self, MockCreator):
        """Successful flow returns PR URL."""
        from agent.actions import ActionExecutor

        creator = MockCreator.return_value
        creator.create_bugfix_pr.return_value = {
            "status": "success",
            "pr_url": "https://github.com/test/repo/pull/1",
        }

        executor = ActionExecutor(flowmap=Mock())
        executor._find_repo_from_diagnosis = Mock(return_value="/tmp/test-repo")
        result = executor._create_bugfix_pr(
            message_ts="123.456",
            channel_id="C123",
            diagnosis="some bug in src/order.ts",
            thread_history=None,
        )

        assert "github.com" in result
