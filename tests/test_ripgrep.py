"""Tests for ripgrep wrapper — JSON output parsing, error handling."""

import json
from flowmap.search.ripgrep import _parse_json_output, is_available


# ---------------------------------------------------------------------------
# JSON output parsing
# ---------------------------------------------------------------------------

_SAMPLE_RG_OUTPUT = """
{"type":"begin","data":{"path":{"text":"/repos/myapp/src/auth.py"}}}
{"type":"match","data":{"path":{"text":"/repos/myapp/src/auth.py"},"lines":{"text":"def validate_token(token: str) -> bool:\\n"},"line_number":42,"absolute_offset":1234,"submatches":[{"match":{"text":"validate_token"},"start":4,"end":18}]}}
{"type":"match","data":{"path":{"text":"/repos/myapp/src/auth.py"},"lines":{"text":"    return validate_token(request.token)\\n"},"line_number":87,"absolute_offset":3456,"submatches":[{"match":{"text":"validate_token"},"start":11,"end":25}]}}
{"type":"end","data":{"path":{"text":"/repos/myapp/src/auth.py"},"binary_offset":null,"stats":{"elapsed":{"secs":0,"nanos":100},"searches":1,"searches_with_match":1,"bytes_searched":5000,"bytes_printed":500,"matched_lines":2,"matches":2}}}
{"type":"begin","data":{"path":{"text":"/repos/myapp/tests/test_auth.py"}}}
{"type":"match","data":{"path":{"text":"/repos/myapp/tests/test_auth.py"},"lines":{"text":"    assert validate_token('abc123')\\n"},"line_number":15,"absolute_offset":200,"submatches":[{"match":{"text":"validate_token"},"start":11,"end":25}]}}
{"type":"end","data":{"path":{"text":"/repos/myapp/tests/test_auth.py"},"binary_offset":null,"stats":{"elapsed":{"secs":0,"nanos":50},"searches":1,"searches_with_match":1,"bytes_searched":1000,"bytes_printed":200,"matched_lines":1,"matches":1}}}
{"data":{"elapsed_total":{"human":"0.001s","nanos":1000000,"secs":0},"stats":{"bytes_printed":700,"bytes_searched":6000,"elapsed":{"human":"0.001s","nanos":1000000,"secs":0},"matched_lines":3,"matches":3,"searches":2,"searches_with_match":2}},"type":"summary"}
""".strip()


def test_parse_basic_matches():
    repo_paths = {"myapp": "/repos/myapp"}
    results = _parse_json_output(_SAMPLE_RG_OUTPUT, repo_paths, limit=50)
    assert len(results) == 3


def test_parse_repo_mapping():
    repo_paths = {"myapp": "/repos/myapp"}
    results = _parse_json_output(_SAMPLE_RG_OUTPUT, repo_paths, limit=50)
    assert all(r.repo == "myapp" for r in results)


def test_parse_relative_paths():
    repo_paths = {"myapp": "/repos/myapp"}
    results = _parse_json_output(_SAMPLE_RG_OUTPUT, repo_paths, limit=50)
    files = {r.file for r in results}
    assert "src/auth.py" in files
    assert "tests/test_auth.py" in files


def test_parse_line_numbers():
    repo_paths = {"myapp": "/repos/myapp"}
    results = _parse_json_output(_SAMPLE_RG_OUTPUT, repo_paths, limit=50)
    lines = [r.line for r in results]
    assert 42 in lines
    assert 87 in lines
    assert 15 in lines


def test_parse_match_text():
    repo_paths = {"myapp": "/repos/myapp"}
    results = _parse_json_output(_SAMPLE_RG_OUTPUT, repo_paths, limit=50)
    assert any("validate_token" in r.text for r in results)


# ---------------------------------------------------------------------------
# Limit enforcement
# ---------------------------------------------------------------------------

def test_limit_enforced():
    repo_paths = {"myapp": "/repos/myapp"}
    results = _parse_json_output(_SAMPLE_RG_OUTPUT, repo_paths, limit=2)
    assert len(results) == 2


def test_limit_zero():
    repo_paths = {"myapp": "/repos/myapp"}
    results = _parse_json_output(_SAMPLE_RG_OUTPUT, repo_paths, limit=0)
    assert len(results) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_output():
    results = _parse_json_output("", {}, limit=50)
    assert results == []


def test_summary_only():
    output = '{"data":{"elapsed_total":{"human":"0.001s"}},"type":"summary"}'
    results = _parse_json_output(output, {}, limit=50)
    assert results == []


def test_malformed_json_lines_skipped():
    output = '{"type":"match","data":{"path":{"text":"/a/b.py"},"lines":{"text":"hello\\n"},"line_number":1,"absolute_offset":0,"submatches":[]}}\nnot valid json\n'
    results = _parse_json_output(output, {"repo": "/a"}, limit=50)
    assert len(results) == 1


def test_multiple_repos():
    """Results from different repos should map to correct repo names."""
    output = (
        '{"type":"begin","data":{"path":{"text":"/code/repo-a/foo.py"}}}\n'
        '{"type":"match","data":{"path":{"text":"/code/repo-a/foo.py"},"lines":{"text":"match1\\n"},"line_number":1,"absolute_offset":0,"submatches":[]}}\n'
        '{"type":"end","data":{"path":{"text":"/code/repo-a/foo.py"},"binary_offset":null,"stats":{"elapsed":{"secs":0,"nanos":0},"searches":1,"searches_with_match":1,"bytes_searched":10,"bytes_printed":10,"matched_lines":1,"matches":1}}}\n'
        '{"type":"begin","data":{"path":{"text":"/code/repo-b/bar.py"}}}\n'
        '{"type":"match","data":{"path":{"text":"/code/repo-b/bar.py"},"lines":{"text":"match2\\n"},"line_number":5,"absolute_offset":0,"submatches":[]}}\n'
        '{"type":"end","data":{"path":{"text":"/code/repo-b/bar.py"},"binary_offset":null,"stats":{"elapsed":{"secs":0,"nanos":0},"searches":1,"searches_with_match":1,"bytes_searched":10,"bytes_printed":10,"matched_lines":1,"matches":1}}}\n'
    )
    repo_paths = {"repo-a": "/code/repo-a", "repo-b": "/code/repo-b"}
    results = _parse_json_output(output, repo_paths, limit=50)
    assert len(results) == 2
    repos = {r.repo for r in results}
    assert repos == {"repo-a", "repo-b"}


def test_is_available_returns_bool():
    """is_available should return a boolean, not raise."""
    result = is_available()
    assert isinstance(result, bool)
