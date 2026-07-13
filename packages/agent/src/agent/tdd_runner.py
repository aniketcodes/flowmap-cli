"""TDDRunner — Detects test frameworks, installs deps, runs tests."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Framework detection patterns
_FRAMEWORK_CONFIGS = {
    "jest": ["jest.config.js", "jest.config.ts", "jest.config.mjs"],
    "pytest": ["pytest.ini", "setup.cfg", "tox.ini"],
    "go": ["go.mod"],
}

_FRAMEWORK_TEST_PATTERNS = {
    "jest": ["*.test.ts", "*.test.js", "*.spec.ts", "*.spec.js"],
    "pytest": ["test_*.py", "*_test.py"],
    "go": ["*_test.go"],
}

_FRAMEWORK_INSTALL_CMDS = {
    "jest": ["npm", "install"],
    "pytest": ["pip", "install", "-e", ".[test]"],
    "go": ["go", "mod", "download"],
}

_FRAMEWORK_RUN_CMDS = {
    "jest": ["npx", "jest", "--no-coverage"],
    "pytest": ["python", "-m", "pytest", "-x", "-q"],
    "go": ["go", "test", "-v"],
}


class TDDRunner:
    """Detects test framework, installs deps, runs tests."""

    def detect_framework(self, repo_path: Path) -> str | None:
        """Detect test framework from repo structure. Returns 'jest', 'pytest', 'go', or None."""
        repo_path = Path(repo_path)

        # 1. Check config files (root level)
        for framework, configs in _FRAMEWORK_CONFIGS.items():
            for config in configs:
                if (repo_path / config).exists():
                    logger.info("detected_framework=%s from config=%s", framework, config)
                    return framework

        # 1b. Check go.mod in subdirectories (monorepo)
        go_mods = list(repo_path.rglob("go.mod"))
        if go_mods:
            logger.info("detected_framework=go from go.mod in subdir")
            return "go"

        # 2. Check for pytest in pyproject.toml
        pyproject = repo_path / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            if "pytest" in content:
                logger.info("detected_framework=pytest from pyproject.toml")
                return "pytest"

        # 3. Check for test files (only if framework has a config hint)
        #    Without config, test file patterns are too ambiguous (e.g., *_test.go could be Go)
        #    So we only use test files as fallback when config detection fails
        for framework, patterns in _FRAMEWORK_TEST_PATTERNS.items():
            # Skip if config already gave us a hint
            if any((repo_path / c).exists() for c in _FRAMEWORK_CONFIGS.get(framework, [])):
                continue
            for pattern in patterns:
                matches = list(repo_path.rglob(pattern))
                if matches:
                    logger.info("detected_framework=%s from pattern=%s matches=%d", framework, pattern, len(matches))
                    return framework

        # 4. Check package.json for jest script
        package_json = repo_path / "package.json"
        if package_json.exists():
            import json
            try:
                data = json.loads(package_json.read_text())
                scripts = data.get("scripts", {})
                test_cmd = scripts.get("test", "")
                if "jest" in test_cmd:
                    logger.info("detected_framework=jest from package.json scripts")
                    return "jest"
            except (json.JSONDecodeError, KeyError):
                pass

        logger.info("no_framework_detected repo=%s", repo_path)
        return None

    def install_deps(self, repo_path: Path, framework: str) -> bool:
        """Install dependencies for the detected framework. Returns True on success."""
        cmd = _FRAMEWORK_INSTALL_CMDS.get(framework)
        if not cmd:
            logger.warning("no_install_command framework=%s", framework)
            return False

        logger.info("installing_deps framework=%s cmd=%s", framework, cmd)
        try:
            result = subprocess.run(
                cmd,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.error("install_failed framework=%s stderr=%s", framework, result.stderr[:500])
                return False
            logger.info("install_success framework=%s", framework)
            return True
        except subprocess.TimeoutExpired:
            logger.error("install_timeout framework=%s", framework)
            return False
        except Exception as e:
            logger.error("install_error framework=%s error=%s", framework, e)
            return False

    def run_test(self, repo_path: Path, test_file: str, framework: str) -> tuple[bool, str]:
        """Run a specific test file. Returns (passed: bool, output: str)."""
        base_cmd = _FRAMEWORK_RUN_CMDS.get(framework)
        if not base_cmd:
            return False, f"Unknown framework: {framework}"

        cmd = base_cmd + [test_file]
        logger.info("running_test framework=%s cmd=%s", framework, cmd)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0
            logger.info("test_result passed=%s output_len=%d", passed, len(output))
            return passed, output
        except subprocess.TimeoutExpired:
            return False, f"Test timed out after 60s"
        except Exception as e:
            return False, f"Test error: {e}"
