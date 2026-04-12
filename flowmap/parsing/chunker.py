"""AST-aware code chunking — the highest-leverage quality improvement in FlowMap."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Node

from flowmap.parsing.languages import get_language_name, get_parser

log = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 8000
FALLBACK_LINES = 80
FALLBACK_OVERLAP = 20

# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    text: str
    chunk_type: str         # function | class | method | preamble | config_block | fallback
    symbol_name: str        # "MyClass.my_method" or ""
    signature: str          # "def foo(x: int) -> str" or ""
    parent_symbol: str      # enclosing class name or ""
    parent_signature: str   # enclosing class declaration line or ""
    start_line: int         # 1-indexed
    end_line: int           # 1-indexed
    language: str


# ---------------------------------------------------------------------------
# Node type definitions per language
# ---------------------------------------------------------------------------

# Top-level definition node types to extract as chunks
_CODE_NODE_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "export_statement",       # captures `export const handler = () => {}`
        "lexical_declaration",    # captures `const handler = () => {}`
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
    },
    "swift": {
        "function_declaration",
        "class_declaration",      # covers class, struct, enum, extension
        "protocol_declaration",
    },
}

# Node types that contain methods/properties (for large class splitting)
_METHOD_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition"},
    "typescript": {
        "method_definition",
        "public_field_definition",
        "field_definition",       # class field declarations
        "property_signature",     # interface property signatures
    },
    "javascript": {"method_definition"},
    "go": set(),  # Go methods are top-level, not inside structs
    "java": {"method_declaration", "constructor_declaration", "field_declaration"},
    "swift": {"function_declaration", "init_declaration", "deinit_declaration", "subscript_declaration", "protocol_function_declaration"},
}

# Node types that are "decorated" wrappers (extract inner definition, not both)
_DECORATED_WRAPPER_TYPES: dict[str, str] = {
    "python": "decorated_definition",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_file(filepath: str, content: str, extension: str) -> list[Chunk]:
    """Chunk a file using the appropriate strategy based on extension.

    Returns a list of Chunk objects. Each chunk is a semantically coherent unit.
    """
    if not content.strip():
        return []

    language = get_language_name(extension)
    parser, strategy = get_parser(extension)

    if strategy == "code" and parser is not None:
        return _chunk_code(content, language, parser)
    elif strategy == "config" and parser is not None:
        return _chunk_config(content, language, parser)
    else:
        return _chunk_fallback(content, language)


# ---------------------------------------------------------------------------
# Code chunking (AST-aware)
# ---------------------------------------------------------------------------


def _chunk_code(content: str, language: str, parser) -> list[Chunk]:
    """Extract functions, classes, methods as individual chunks. Collect orphans as preamble."""
    content_bytes = content.encode("utf-8")
    tree = parser.parse(content_bytes)
    root = tree.root_node

    code_types = _CODE_NODE_TYPES.get(language, set())
    decorated_type = _DECORATED_WRAPPER_TYPES.get(language)

    chunks: list[Chunk] = []
    extracted_ranges: list[tuple[int, int]] = []  # (start_byte, end_byte) of extracted nodes

    for child in root.children:
        node_type = child.type

        # CommonJS exports (JS/TS only) — handled before code_types gate
        if node_type == "expression_statement" and language in ("javascript", "typescript"):
            cjs = _find_commonjs_export(child, content_bytes)
            if cjs is not None:
                value_node, cjs_name = cjs
                chunk = _node_to_chunk(child, content_bytes, language, inner_def=value_node)
                if chunk:
                    chunk.symbol_name = cjs_name
                    chunks.append(chunk)
                    extracted_ranges.append((child.start_byte, child.end_byte))
            continue  # expression_statements: CJS ones extracted above, rest stay in preamble

        if node_type not in code_types:
            continue

        inner = None  # track inner definition for export/decorated nodes

        # Handle decorated definitions — extract the outer node, skip inner
        if decorated_type and node_type == decorated_type:
            inner = _find_inner_definition(child)
            if inner:
                chunk = _node_to_chunk(child, content_bytes, language, inner_def=inner)
            else:
                chunk = _node_to_chunk(child, content_bytes, language)
        # Handle export statements — extract the inner declaration
        elif node_type in ("export_statement", "lexical_declaration"):
            inner = _find_exportable_definition(child)
            if inner is None:
                continue  # skip exports without a meaningful definition
            chunk = _node_to_chunk(child, content_bytes, language, inner_def=inner)
        else:
            chunk = _node_to_chunk(child, content_bytes, language)

        if chunk:
            # Split large classes into per-method chunks
            if chunk.chunk_type == "class" and len(chunk.text) > MAX_CHUNK_CHARS:
                # For export statements, pass the inner class node to _split_class
                # (export_statement has no "name" or "body" fields — the class inside it does)
                split_node = child
                if child.type in ("export_statement", "lexical_declaration") and inner is not None:
                    # inner was set by _find_exportable_definition above
                    if inner.type in ("class_declaration", "class_definition"):
                        split_node = inner
                method_chunks = _split_class(split_node, content_bytes, language)
                if method_chunks:
                    chunks.extend(method_chunks)
                    extracted_ranges.append((child.start_byte, child.end_byte))
                    continue

            chunks.append(chunk)
            extracted_ranges.append((child.start_byte, child.end_byte))

    # Collect preamble — all top-level text NOT inside extracted nodes
    preamble = _extract_preamble(content_bytes, extracted_ranges, language)
    if preamble:
        # Split large preambles
        if len(preamble.text) > MAX_CHUNK_CHARS:
            chunks.extend(_split_preamble(preamble))
        else:
            chunks.append(preamble)

    return chunks


def _node_to_chunk(
    node: Node, content_bytes: bytes, language: str, inner_def: Node | None = None,
) -> Chunk | None:
    """Convert a tree-sitter node to a Chunk."""
    text = content_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    if not text.strip():
        return None

    # Determine chunk type and symbol name
    target = inner_def or node
    chunk_type, symbol_name = _extract_symbol_info(target, content_bytes, language)

    # Extract signature (first line up to opening brace/colon/body)
    signature = _extract_signature(text)

    # Extract parent context
    parent_symbol, parent_signature = _extract_parent_context(node, content_bytes)

    return Chunk(
        text=text,
        chunk_type=chunk_type,
        symbol_name=symbol_name,
        signature=signature,
        parent_symbol=parent_symbol,
        parent_signature=parent_signature,
        start_line=node.start_point[0] + 1,  # tree-sitter is 0-indexed
        end_line=node.end_point[0] + 1,
        language=language,
    )


def _extract_symbol_info(node: Node, content_bytes: bytes, language: str) -> tuple[str, str]:
    """Extract (chunk_type, symbol_name) from a tree-sitter node."""
    node_type = node.type

    # Map node types to chunk types
    type_map = {
        "function_definition": "function",
        "function_declaration": "function",
        "class_definition": "class",
        "class_declaration": "class",
        "method_definition": "method",
        "method_declaration": "method",
        "constructor_declaration": "method",
        "interface_declaration": "class",
        "type_alias_declaration": "class",
        "enum_declaration": "class",
        "record_declaration": "class",
        "type_declaration": "class",
        "arrow_function": "function",
        "public_field_definition": "property",
        "field_definition": "property",
        "property_signature": "property",
        "field_declaration": "property",
        "protocol_declaration": "class",
        "init_declaration": "method",
        "deinit_declaration": "method",
        "subscript_declaration": "method",
        "protocol_function_declaration": "method",
        "class": "class",                # JS: module.exports = class Service {}
        "function_expression": "function",
        "generator_function": "function",
    }
    chunk_type = type_map.get(node_type, "function")

    # Extract symbol name from the 'name' child
    symbol_name = ""
    name_node = node.child_by_field_name("name")
    if name_node:
        symbol_name = content_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")

    # Swift deinit has no name field
    if not symbol_name and node_type == "deinit_declaration":
        symbol_name = "deinit"

    # Go method receiver: extract type name for qualified symbol
    if language == "go" and node_type == "method_declaration":
        receiver = node.child_by_field_name("receiver")
        if receiver:
            recv_type = _extract_go_receiver_type(receiver, content_bytes)
            if recv_type and symbol_name:
                symbol_name = f"{recv_type}.{symbol_name}"
                chunk_type = "method"

    return chunk_type, symbol_name


def _extract_go_receiver_type(receiver_node: Node, content_bytes: bytes) -> str:
    """Extract the type name from a Go method receiver, stripping pointer prefix."""
    # Walk into parameter_list -> parameter_declaration -> type
    for child in receiver_node.children:
        if child.type == "parameter_declaration":
            type_node = child.child_by_field_name("type")
            if type_node:
                type_text = content_bytes[type_node.start_byte:type_node.end_byte].decode("utf-8", errors="replace")
                return type_text.lstrip("*")
    return ""


def _extract_signature(text: str) -> str:
    """Extract the signature line from chunk text (first line up to body start)."""
    first_line = text.split("\n")[0].rstrip()
    # Truncate overly long signatures
    if len(first_line) > 200:
        first_line = first_line[:200] + "..."
    return first_line


def _extract_parent_context(node: Node, content_bytes: bytes) -> tuple[str, str]:
    """Walk up the AST to find enclosing class/module context."""
    parent = node.parent
    while parent:
        if parent.type in (
            "class_definition", "class_declaration",
            "interface_declaration", "protocol_declaration",
        ):
            name_node = parent.child_by_field_name("name")
            parent_symbol = content_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace") if name_node else ""
            # Only decode up to first newline for performance (avoids decoding entire large class)
            first_nl = content_bytes.find(b"\n", parent.start_byte)
            if first_nl == -1 or first_nl > parent.end_byte:
                first_nl = parent.end_byte
            parent_sig = content_bytes[parent.start_byte:first_nl].decode("utf-8", errors="replace").rstrip()
            if len(parent_sig) > 200:
                parent_sig = parent_sig[:200] + "..."
            return parent_symbol, parent_sig
        parent = parent.parent
    return "", ""


def _find_inner_definition(decorated_node: Node) -> Node | None:
    """Find the actual function/class inside a decorated_definition."""
    for child in decorated_node.children:
        if child.type in ("function_definition", "class_definition", "decorated_definition"):
            if child.type == "decorated_definition":
                return _find_inner_definition(child)
            return child
    return None


def _find_exportable_definition(node: Node) -> Node | None:
    """Find a meaningful definition inside an export_statement or lexical_declaration.

    Targets: `export function`, `export class`, `export const x = () => {}`,
    `const handler = async () => {}`.
    """
    for child in node.children:
        if child.type in (
            "function_declaration", "class_declaration",
            "interface_declaration", "type_alias_declaration",
            "enum_declaration",
        ):
            return child
        if child.type == "lexical_declaration":
            return _find_exportable_definition(child)
        if child.type == "variable_declarator":
            # const handler = () => {} or const handler = async () => {}
            value = child.child_by_field_name("value")
            if value and value.type in ("arrow_function", "function"):
                return child
        # export default () => {} — anonymous arrow function directly in export
        if child.type in ("arrow_function", "function"):
            return child
    return None


# Accepted right-side types for CommonJS exports
_CJS_VALUE_TYPES = {"function_expression", "arrow_function", "function", "class", "generator_function"}


def _find_commonjs_export(node: Node, content_bytes: bytes) -> tuple[Node, str] | None:
    """Detect CommonJS export patterns in an expression_statement.

    Returns (value_node, symbol_name) or None.
    Patterns: module.exports.X = fn, exports.X = fn, module.exports = fn/class/object
    """
    for child in node.children:
        if child.type != "assignment_expression":
            continue

        left = child.child_by_field_name("left")
        right = child.child_by_field_name("right")
        if left is None or right is None or left.type != "member_expression":
            return None

        left_obj = left.child_by_field_name("object")
        left_prop = left.child_by_field_name("property")
        if left_obj is None or left_prop is None:
            return None

        obj_text = content_bytes[left_obj.start_byte:left_obj.end_byte].decode("utf-8", errors="replace")
        prop_text = content_bytes[left_prop.start_byte:left_prop.end_byte].decode("utf-8", errors="replace")

        # Pattern: exports.X = fn
        if left_obj.type == "identifier" and obj_text == "exports":
            if right.type in _CJS_VALUE_TYPES:
                return right, prop_text
            return None

        # Pattern: module.exports.X = fn  OR  module.exports = fn
        if left_obj.type == "member_expression":
            # left is module.exports.X — left_obj is module.exports, left_prop is X
            inner_obj = left_obj.child_by_field_name("object")
            inner_prop = left_obj.child_by_field_name("property")
            if inner_obj is None or inner_prop is None:
                return None
            inner_obj_text = content_bytes[inner_obj.start_byte:inner_obj.end_byte].decode("utf-8", errors="replace")
            inner_prop_text = content_bytes[inner_prop.start_byte:inner_prop.end_byte].decode("utf-8", errors="replace")
            if inner_obj_text == "module" and inner_prop_text == "exports":
                if right.type in _CJS_VALUE_TYPES:
                    return right, prop_text
            return None

        if left_obj.type == "identifier" and obj_text == "module" and prop_text == "exports":
            # Pattern: module.exports = fn/class/object
            if right.type in _CJS_VALUE_TYPES:
                # Prefer the function/class name if it has one
                name_node = right.child_by_field_name("name")
                name = content_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace") if name_node else "module.exports"
                return right, name
            if right.type == "object":
                # Bulk export: module.exports = { ... } — single chunk
                return right, "module.exports"
            return None

        return None
    return None


# ---------------------------------------------------------------------------
# Class splitting
# ---------------------------------------------------------------------------


def _split_class(class_node: Node, content_bytes: bytes, language: str) -> list[Chunk]:
    """Split a large class into signature chunk + per-method chunks."""
    method_types = _METHOD_NODE_TYPES.get(language, set())
    if not method_types:
        return []

    class_text = content_bytes[class_node.start_byte:class_node.end_byte].decode("utf-8", errors="replace")
    class_name = ""
    name_node = class_node.child_by_field_name("name")
    if name_node:
        class_name = content_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")

    class_sig = class_text.split("\n")[0].rstrip()
    if len(class_sig) > 200:
        class_sig = class_sig[:200] + "..."

    chunks: list[Chunk] = []

    # Signature chunk (class declaration line + docstring if present)
    sig_text = class_sig
    for child in class_node.children:
        if child.type in ("block", "class_body", "enum_class_body", "protocol_body", "{"):
            break
        if child.type in ("expression_statement", "comment"):
            sig_text += "\n" + content_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")

    chunks.append(Chunk(
        text=sig_text,
        chunk_type="class",
        symbol_name=class_name,
        signature=class_sig,
        parent_symbol="",
        parent_signature="",
        start_line=class_node.start_point[0] + 1,
        end_line=class_node.start_point[0] + 1 + sig_text.count("\n"),
        language=language,
    ))

    # Per-method chunks
    body = class_node.child_by_field_name("body")
    if body is None:
        # Try finding the body block by type
        for child in class_node.children:
            if child.type in ("block", "class_body", "enum_class_body", "protocol_body"):
                body = child
                break

    if body:
        for child in body.children:
            if child.type in method_types or (
                child.type == "decorated_definition" and language == "python"
            ):
                # For decorated methods, find inner def so symbol name is extracted correctly
                inner = None
                if child.type == "decorated_definition":
                    inner = _find_inner_definition(child)
                chunk = _node_to_chunk(child, content_bytes, language, inner_def=inner)
                if chunk:
                    chunk.parent_symbol = class_name
                    chunk.parent_signature = class_sig
                    if chunk.symbol_name and class_name:
                        chunk.symbol_name = f"{class_name}.{chunk.symbol_name}"
                    chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Preamble extraction
# ---------------------------------------------------------------------------


def _extract_preamble(content_bytes: bytes, extracted_ranges: list[tuple[int, int]], language: str) -> Chunk | None:
    """Collect all text NOT inside extracted nodes as a preamble chunk."""
    content = content_bytes.decode("utf-8", errors="replace")
    if not extracted_ranges:
        return Chunk(
            text=content,
            chunk_type="preamble",
            symbol_name="",
            signature="",
            parent_symbol="",
            parent_signature="",
            start_line=1,
            end_line=content.count("\n") + 1,
            language=language,
        )

    # Sort ranges by start position
    ranges = sorted(extracted_ranges)
    preamble_parts: list[str] = []
    first_line = None  # track actual source line numbers
    last_line = None

    # Helper: count newlines before a byte offset to get line number
    def _byte_to_line(offset: int) -> int:
        return content_bytes[:offset].count(b"\n") + 1

    # Text before first extracted node
    if ranges[0][0] > 0:
        text = content_bytes[:ranges[0][0]].decode("utf-8", errors="replace")
        if text.strip():
            preamble_parts.append(text)
            first_line = 1
            last_line = _byte_to_line(ranges[0][0])

    # Gaps between extracted nodes
    for i in range(len(ranges) - 1):
        gap = content_bytes[ranges[i][1]:ranges[i + 1][0]].decode("utf-8", errors="replace")
        if gap.strip():
            preamble_parts.append(gap)
            gap_start = _byte_to_line(ranges[i][1])
            gap_end = _byte_to_line(ranges[i + 1][0])
            if first_line is None:
                first_line = gap_start
            last_line = gap_end

    # Text after last extracted node
    if ranges[-1][1] < len(content_bytes):
        tail = content_bytes[ranges[-1][1]:].decode("utf-8", errors="replace")
        if tail.strip():
            preamble_parts.append(tail)
            tail_start = _byte_to_line(ranges[-1][1])
            tail_end = _byte_to_line(len(content_bytes))
            if first_line is None:
                first_line = tail_start
            last_line = tail_end

    preamble_text = "\n".join(part.strip() for part in preamble_parts if part.strip())
    if not preamble_text.strip():
        return None

    return Chunk(
        text=preamble_text,
        chunk_type="preamble",
        symbol_name="",
        signature="",
        parent_symbol="",
        parent_signature="",
        start_line=first_line or 1,
        end_line=last_line or (preamble_text.count("\n") + 1),
        language=language,
    )


def _split_preamble(preamble: Chunk) -> list[Chunk]:
    """Split an oversized preamble at blank-line boundaries."""
    lines = preamble.text.splitlines(keepends=True)
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_start = preamble.start_line

    for i, line in enumerate(lines):
        current_lines.append(line)
        text_so_far = "".join(current_lines)

        # Split at blank lines when approaching the limit
        if len(text_so_far) > MAX_CHUNK_CHARS and line.strip() == "":
            chunks.append(Chunk(
                text=text_so_far.rstrip(),
                chunk_type="preamble",
                symbol_name="",
                signature="",
                parent_symbol=preamble.parent_symbol,
                parent_signature=preamble.parent_signature,
                start_line=current_start,
                end_line=current_start + len(current_lines) - 1,
                language=preamble.language,
            ))
            current_start = current_start + len(current_lines)
            current_lines = []

    # Remaining lines
    if current_lines:
        chunks.append(Chunk(
            text="".join(current_lines).rstrip(),
            chunk_type="preamble",
            symbol_name="",
            signature="",
            parent_symbol=preamble.parent_symbol,
            parent_signature=preamble.parent_signature,
            start_line=current_start,
            end_line=current_start + len(current_lines) - 1,
            language=preamble.language,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Config file chunking
# ---------------------------------------------------------------------------


def _chunk_config(content: str, language: str, parser) -> list[Chunk]:
    """Chunk config files (YAML, JSON) by top-level keys."""
    content_bytes = content.encode("utf-8")
    tree = parser.parse(content_bytes)
    root = tree.root_node

    chunks: list[Chunk] = []

    if language == "yaml":
        # stream > document > block_node > block_mapping > block_mapping_pair (depth 4)
        for node in _walk_children(root, max_depth=5):
            if node.type == "block_mapping_pair" and _is_top_level_yaml(node):
                text = content_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                key_node = node.child_by_field_name("key")
                key_name = content_bytes[key_node.start_byte:key_node.end_byte].decode("utf-8", errors="replace") if key_node else ""
                chunks.append(Chunk(
                    text=text,
                    chunk_type="config_block",
                    symbol_name=key_name,
                    signature=text.split("\n")[0].rstrip(),
                    parent_symbol="",
                    parent_signature="",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    language=language,
                ))
    elif language == "json":
        # document > object > pair (depth 2)
        for node in _walk_children(root, max_depth=3):
            if node.type == "pair" and _is_top_level_json(node):
                text = content_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                key_node = node.child_by_field_name("key")
                key_name = content_bytes[key_node.start_byte:key_node.end_byte].decode("utf-8", errors="replace").strip('"') if key_node else ""
                chunks.append(Chunk(
                    text=text,
                    chunk_type="config_block",
                    symbol_name=key_name,
                    signature=text.split("\n")[0].rstrip(),
                    parent_symbol="",
                    parent_signature="",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    language=language,
                ))

    # Fallback if no config blocks found
    if not chunks:
        return _chunk_fallback(content, language)

    return chunks


def _is_top_level_yaml(node: Node) -> bool:
    """Check if a YAML block_mapping_pair is at the top level (not nested)."""
    # Walk up: block_mapping_pair -> block_mapping -> block_node -> ... -> stream
    parent = node.parent
    depth = 0
    while parent:
        if parent.type == "block_mapping_pair":
            return False  # nested inside another mapping pair
        parent = parent.parent
        depth += 1
    return True


def _is_top_level_json(node: Node) -> bool:
    """Check if a JSON pair is a direct child of the root object."""
    parent = node.parent
    if parent and parent.type == "object":
        grandparent = parent.parent
        if grandparent and grandparent.type == "document":
            return True
    return False


def _walk_children(node: Node, max_depth: int = 3, depth: int = 0):
    """Yield children up to max_depth. Avoids deep recursion into nested structures."""
    if depth > max_depth:
        return
    for child in node.children:
        yield child
        yield from _walk_children(child, max_depth, depth + 1)


# ---------------------------------------------------------------------------
# Fallback chunking (line-based sliding window)
# ---------------------------------------------------------------------------


def _chunk_fallback(content: str, language: str) -> list[Chunk]:
    """Line-based sliding window with blank-line preference."""
    lines = content.splitlines(keepends=True)
    if not lines:
        return []

    if len(lines) <= FALLBACK_LINES:
        return [Chunk(
            text=content,
            chunk_type="fallback",
            symbol_name="",
            signature="",
            parent_symbol="",
            parent_signature="",
            start_line=1,
            end_line=len(lines),
            language=language,
        )]

    chunks: list[Chunk] = []
    start = 0

    while start < len(lines):
        end = min(start + FALLBACK_LINES, len(lines))

        # Try to break at a blank line near the end
        if end < len(lines):
            for i in range(end, max(start + FALLBACK_LINES // 2, start), -1):
                if i < len(lines) and lines[i].strip() == "":
                    end = i + 1
                    break

        chunk_text = "".join(lines[start:end]).rstrip()
        if chunk_text.strip():
            chunks.append(Chunk(
                text=chunk_text,
                chunk_type="fallback",
                symbol_name="",
                signature="",
                parent_symbol="",
                parent_signature="",
                start_line=start + 1,
                end_line=end,
                language=language,
            ))

        start = end - FALLBACK_OVERLAP
        if start >= len(lines) - FALLBACK_OVERLAP:
            # Capture any remaining tail lines not yet in a chunk
            if end < len(lines):
                tail_text = "".join(lines[end:]).rstrip()
                if tail_text.strip():
                    chunks.append(Chunk(
                        text=tail_text,
                        chunk_type="fallback",
                        symbol_name="",
                        signature="",
                        parent_symbol="",
                        parent_signature="",
                        start_line=end + 1,
                        end_line=len(lines),
                        language=language,
                    ))
            break

    return chunks
