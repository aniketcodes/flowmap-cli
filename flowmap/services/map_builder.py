"""Map builder — transform raw store rows into structured repo map data."""

from __future__ import annotations

from collections import defaultdict


def build_repo_map(rows: list[dict]) -> list[dict]:
    """Transform raw store rows into structured repo map output.

    Groups by repo, nests methods under classes, extracts functions.
    Returns list of repo dicts ready for rendering.
    """
    repos_data: dict[str, dict] = {}
    for r in rows:
        rname = r.get("repo", "")
        if rname not in repos_data:
            repos_data[rname] = {
                "name": rname,
                "files": set(),
                "languages": defaultdict(int),
                "classes": [],
                "functions": [],
                "methods": [],
            }
        rd = repos_data[rname]
        rd["files"].add(r.get("file", ""))
        lang = r.get("language", "")
        if lang:
            rd["languages"][lang] += 1

        sym = r.get("symbol_name", "")
        ctype = r.get("chunk_type", "")
        if not sym:
            continue

        entry = {
            "name": sym,
            "file": r.get("file", ""),
            "line": r.get("start_line", 0),
            "signature": r.get("signature", ""),
        }

        if ctype == "class":
            rd["classes"].append(entry)
        elif ctype == "method":
            rd["methods"].append(entry)
        elif ctype == "function":
            parent = r.get("parent_symbol", "")
            if parent:
                entry["parent"] = parent
                rd["methods"].append(entry)
            else:
                rd["functions"].append(entry)

    output_repos = []
    for rname, rd in sorted(repos_data.items()):
        class_methods: dict[str, list] = defaultdict(list)
        for m in rd["methods"]:
            parent = m.get("parent", "")
            if not parent and "." in m["name"]:
                parent = m["name"].rsplit(".", 1)[0]
            if parent:
                class_methods[parent].append(m["name"].split(".")[-1] if "." in m["name"] else m["name"])

        classes_out = []
        for c in rd["classes"]:
            classes_out.append({
                "name": c["name"],
                "file": c["file"],
                "line": c["line"],
                "methods": class_methods.get(c["name"], []),
            })

        repo_out = {
            "name": rname,
            "files": len(rd["files"]),
            "languages": dict(rd["languages"]),
            "classes": classes_out,
            "functions": [{"name": f["name"], "file": f["file"], "line": f["line"], "signature": f["signature"]} for f in rd["functions"]],
        }
        output_repos.append(repo_out)

    return output_repos
