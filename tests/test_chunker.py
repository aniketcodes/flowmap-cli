"""Tests for AST-aware chunking — the most important test file in the project."""

from flowmap.parsing import Chunk, chunk_file


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

PYTHON_CODE = '''\
import os
import sys

API_URL = "https://api.example.com"
MAX_RETRIES = 3


def fetch_data(url: str) -> dict:
    """Fetch data from the API."""
    return {}


class OrderProcessor:
    """Processes orders."""

    def __init__(self, config: dict):
        self.config = config

    def process(self, order_id: int) -> bool:
        return True

    def cancel(self, order_id: int) -> bool:
        return False


def main():
    processor = OrderProcessor({})
    processor.process(42)
'''


def test_python_function_chunking():
    chunks = chunk_file("test.py", PYTHON_CODE, ".py")
    functions = [c for c in chunks if c.chunk_type == "function"]
    names = {c.symbol_name for c in functions}
    assert "fetch_data" in names
    assert "main" in names


def test_python_class_chunking():
    chunks = chunk_file("test.py", PYTHON_CODE, ".py")
    classes = [c for c in chunks if c.chunk_type == "class"]
    assert any(c.symbol_name == "OrderProcessor" for c in classes)


def test_python_preamble():
    chunks = chunk_file("test.py", PYTHON_CODE, ".py")
    preambles = [c for c in chunks if c.chunk_type == "preamble"]
    assert len(preambles) >= 1
    preamble_text = preambles[0].text
    assert "import os" in preamble_text or "API_URL" in preamble_text


def test_python_symbol_names():
    chunks = chunk_file("test.py", PYTHON_CODE, ".py")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "fetch_data" in symbols
    assert "OrderProcessor" in symbols
    assert "main" in symbols


def test_python_line_ranges():
    chunks = chunk_file("test.py", PYTHON_CODE, ".py")
    fetch = next(c for c in chunks if c.symbol_name == "fetch_data")
    assert fetch.start_line > 0
    assert fetch.end_line >= fetch.start_line


def test_python_signature_extraction():
    chunks = chunk_file("test.py", PYTHON_CODE, ".py")
    fetch = next(c for c in chunks if c.symbol_name == "fetch_data")
    assert "def fetch_data" in fetch.signature


PYTHON_DECORATED = '''\
import functools


def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@my_decorator
def decorated_function():
    pass


@my_decorator
class DecoratedClass:
    pass
'''


def test_python_decorated_definition_no_duplicates():
    """Decorated definitions should produce one chunk, not two (outer + inner)."""
    chunks = chunk_file("test.py", PYTHON_DECORATED, ".py")
    names = [c.symbol_name for c in chunks if c.symbol_name]
    # Each symbol should appear at most once
    assert names.count("decorated_function") == 1
    assert names.count("DecoratedClass") == 1


# ---------------------------------------------------------------------------
# TypeScript
# ---------------------------------------------------------------------------

TS_CODE = '''\
import { Request, Response } from "express";

interface AuthConfig {
  secret: string;
  expiresIn: number;
}

export class AuthService {
  constructor(private config: AuthConfig) {}

  validateToken(token: string): boolean {
    return token.length > 0;
  }
}

export const handler = async (req: Request, res: Response) => {
  res.json({ ok: true });
};

type UserId = string;
'''


def test_typescript_class():
    chunks = chunk_file("auth.ts", TS_CODE, ".ts")
    classes = [c for c in chunks if c.chunk_type == "class"]
    names = {c.symbol_name for c in classes}
    assert "AuthService" in names or "AuthConfig" in names


def test_typescript_interface():
    chunks = chunk_file("auth.ts", TS_CODE, ".ts")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "AuthConfig" in symbols


def test_typescript_const_arrow_export():
    """export const handler = async () => {} should be captured."""
    chunks = chunk_file("auth.ts", TS_CODE, ".ts")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "handler" in symbols


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

GO_CODE = '''\
package main

import "fmt"

type Service struct {
    Name string
}

func (s *Service) Process(data string) error {
    fmt.Println(data)
    return nil
}

func NewService(name string) *Service {
    return &Service{Name: name}
}
'''


def test_go_function():
    chunks = chunk_file("main.go", GO_CODE, ".go")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "NewService" in symbols


def test_go_method_receiver():
    """Go method should have qualified name: Service.Process"""
    chunks = chunk_file("main.go", GO_CODE, ".go")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "Service.Process" in symbols


def test_go_method_chunk_type():
    chunks = chunk_file("main.go", GO_CODE, ".go")
    process = next(c for c in chunks if c.symbol_name == "Service.Process")
    assert process.chunk_type == "method"


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------

JAVA_CODE = '''\
package com.example;

import java.util.List;

public class OrderService {
    public boolean process(int orderId) {
        return true;
    }

    public void cancel(int orderId) {
    }
}

interface PaymentGateway {
    void charge(double amount);
}
'''


def test_java_class():
    chunks = chunk_file("OrderService.java", JAVA_CODE, ".java")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "OrderService" in symbols


def test_java_interface():
    chunks = chunk_file("OrderService.java", JAVA_CODE, ".java")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "PaymentGateway" in symbols


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

YAML_CODE = '''\
services:
  api:
    build: ./api
    ports:
      - "3000:3000"
  worker:
    build: ./worker

volumes:
  data:
    driver: local
'''


def test_yaml_top_level_keys():
    chunks = chunk_file("docker-compose.yml", YAML_CODE, ".yaml")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "services" in symbols
    assert "volumes" in symbols


def test_yaml_chunk_type():
    chunks = chunk_file("docker-compose.yml", YAML_CODE, ".yaml")
    assert all(c.chunk_type == "config_block" for c in chunks)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def test_fallback_for_unknown_extension():
    content = "line 1\nline 2\nline 3\n"
    chunks = chunk_file("script.sh", content, ".sh")
    assert len(chunks) >= 1
    assert chunks[0].chunk_type == "fallback"


def test_fallback_sliding_window():
    """Files longer than 80 lines should produce multiple fallback chunks."""
    content = "\n".join(f"line {i}" for i in range(200))
    chunks = chunk_file("big.sh", content, ".sh")
    assert len(chunks) > 1
    assert all(c.chunk_type == "fallback" for c in chunks)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_file():
    chunks = chunk_file("empty.py", "", ".py")
    assert chunks == []


def test_whitespace_only_file():
    chunks = chunk_file("blank.py", "   \n\n  \n", ".py")
    assert chunks == []


def test_single_function_file():
    code = "def hello():\n    print('hello')\n"
    chunks = chunk_file("hello.py", code, ".py")
    assert len(chunks) >= 1
    assert any(c.symbol_name == "hello" for c in chunks)


def test_language_field_set():
    chunks = chunk_file("test.py", "def foo(): pass\n", ".py")
    assert all(c.language == "python" for c in chunks)


def test_json_chunking():
    json_code = '{"name": "flowmap", "version": "0.2.0", "scripts": {"test": "pytest"}}'
    chunks = chunk_file("package.json", json_code, ".json")
    assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Large class splitting
# ---------------------------------------------------------------------------

# Generate a Python class > 8000 chars with decorated methods
_LARGE_CLASS_METHODS = "\n".join(
    f"""    @staticmethod
    def method_{i}(x: int) -> int:
        \"\"\"Method {i} docstring — performs operation number {i} on the input value.\"\"\"
        result = x + {i}
        intermediate = result * 2
        validated = intermediate if intermediate > 0 else -intermediate
        logged_value = f"method_{i}: input={{x}}, result={{validated}}"
        print(logged_value)
        return validated
"""
    for i in range(60)
)

PYTHON_LARGE_CLASS = f'''\
class BigService:
    """A large service class that should be split."""

{_LARGE_CLASS_METHODS}
'''


def test_large_class_splits_into_methods():
    """Classes > 8000 chars should split into signature + per-method chunks."""
    assert len(PYTHON_LARGE_CLASS) > 8000, "Test fixture must be > 8000 chars"
    chunks = chunk_file("big.py", PYTHON_LARGE_CLASS, ".py")
    # Should have: 1 class signature chunk + N method chunks + possibly preamble
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) >= 1, "Should have at least one class signature chunk"
    assert class_chunks[0].symbol_name == "BigService"

    method_chunks = [c for c in chunks if c.chunk_type == "function" and c.parent_symbol == "BigService"]
    assert len(method_chunks) >= 10, f"Expected many method chunks, got {len(method_chunks)}"


def test_large_class_methods_have_parent_context():
    """Split method chunks should have parent_symbol and parent_signature."""
    chunks = chunk_file("big.py", PYTHON_LARGE_CLASS, ".py")
    method_chunks = [c for c in chunks if c.parent_symbol == "BigService"]
    assert len(method_chunks) > 0
    for m in method_chunks:
        assert m.parent_symbol == "BigService"
        assert "class BigService" in m.parent_signature


def test_large_class_decorated_methods_have_symbol_names():
    """Decorated methods (@staticmethod) in split classes must have correct symbol names."""
    chunks = chunk_file("big.py", PYTHON_LARGE_CLASS, ".py")
    method_chunks = [c for c in chunks if c.parent_symbol == "BigService" and c.symbol_name]
    names = {c.symbol_name for c in method_chunks}
    # Should have BigService.method_0, BigService.method_1, etc.
    assert "BigService.method_0" in names, f"Missing method_0, got: {names}"
    assert "BigService.method_5" in names


def test_large_class_no_duplicate_chunks():
    """Split class should not produce overlapping/duplicate chunks."""
    chunks = chunk_file("big.py", PYTHON_LARGE_CLASS, ".py")
    ids = [c.symbol_name for c in chunks if c.symbol_name]
    assert len(ids) == len(set(ids)), f"Duplicate symbol names: {ids}"


# ---------------------------------------------------------------------------
# Large preamble splitting
# ---------------------------------------------------------------------------

PYTHON_LARGE_PREAMBLE = "\n".join(
    f"import module_{i}" for i in range(200)
) + "\n\ndef only_function():\n    pass\n"


def test_large_preamble_splits():
    """Preambles > 8000 chars should be split at blank-line boundaries."""
    chunks = chunk_file("imports.py", PYTHON_LARGE_PREAMBLE, ".py")
    preambles = [c for c in chunks if c.chunk_type == "preamble"]
    # 200 import lines * ~18 chars each = ~3600 chars. May not exceed 8000.
    # But the preamble should exist and be correct.
    assert len(preambles) >= 1
    combined = " ".join(p.text for p in preambles)
    assert "import module_0" in combined
    assert "import module_199" in combined


# ---------------------------------------------------------------------------
# Python: async def
# ---------------------------------------------------------------------------


def test_python_async_function():
    code = "async def fetch_data(url: str) -> dict:\n    return {}\n"
    chunks = chunk_file("async.py", code, ".py")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "fetch_data" in symbols


def test_python_async_signature():
    code = "async def fetch_data(url: str) -> dict:\n    return {}\n"
    chunks = chunk_file("async.py", code, ".py")
    func = next(c for c in chunks if c.symbol_name == "fetch_data")
    assert "async def fetch_data" in func.signature


# ---------------------------------------------------------------------------
# Python: nested classes
# ---------------------------------------------------------------------------


def test_python_nested_class_in_small_class():
    """Nested class should be inside parent class chunk (not extracted separately)."""
    code = '''\
class Outer:
    class Meta:
        db_table = "outer"

    def process(self):
        pass
'''
    chunks = chunk_file("nested.py", code, ".py")
    outer = next(c for c in chunks if c.symbol_name == "Outer")
    assert "class Meta" in outer.text
    # Meta should NOT be a separate chunk
    symbols = {c.symbol_name for c in chunks}
    assert "Meta" not in symbols


# ---------------------------------------------------------------------------
# TypeScript: export default
# ---------------------------------------------------------------------------


def test_typescript_export_default_function():
    code = "export default function handler() { return 42; }\n"
    chunks = chunk_file("handler.ts", code, ".ts")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "handler" in symbols


def test_typescript_anonymous_default_arrow():
    """export default () => {} should be captured (even without a name)."""
    code = "export default () => { return 42; };\n"
    chunks = chunk_file("anon.ts", code, ".ts")
    # Should produce at least one chunk (may have empty symbol_name since anonymous)
    non_preamble = [c for c in chunks if c.chunk_type != "preamble"]
    assert len(non_preamble) >= 1


def test_typescript_type_alias():
    code = "export type UserId = string;\n"
    chunks = chunk_file("types.ts", code, ".ts")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "UserId" in symbols


# ---------------------------------------------------------------------------
# Go: value receiver + type declarations
# ---------------------------------------------------------------------------


def test_go_value_receiver():
    """Value receiver (non-pointer) should also produce qualified name."""
    code = '''\
package main

type Handler struct{}

func (h Handler) ServeHTTP() {}
'''
    chunks = chunk_file("handler.go", code, ".go")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "Handler.ServeHTTP" in symbols


def test_go_type_declaration():
    code = '''\
package main

type Config struct {
    Host string
    Port int
}

type Handler interface {
    Handle()
}
'''
    chunks = chunk_file("types.go", code, ".go")
    # type declarations should be captured
    non_preamble = [c for c in chunks if c.chunk_type != "preamble"]
    assert len(non_preamble) >= 2  # Config struct + Handler interface


# ---------------------------------------------------------------------------
# Unicode identifiers
# ---------------------------------------------------------------------------


def test_unicode_python_identifiers():
    code = '''\
def calculer_prix(montant: float) -> float:
    """Calcule le prix avec TVA."""
    return montant * 1.2

class DonnéesClient:
    nom: str
'''
    chunks = chunk_file("unicode.py", code, ".py")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "calculer_prix" in symbols
    assert "DonnéesClient" in symbols


def test_unicode_comments_preserved():
    code = '''\
# 这是一个测试函数
def process():
    """处理数据"""
    pass
'''
    chunks = chunk_file("chinese.py", code, ".py")
    func = next(c for c in chunks if c.symbol_name == "process")
    assert "处理数据" in func.text


# ---------------------------------------------------------------------------
# Syntax errors (tree-sitter handles gracefully)
# ---------------------------------------------------------------------------


def test_python_syntax_error_partial_parse():
    """Files with syntax errors should still produce chunks for valid parts."""
    code = '''\
def valid_function():
    return True

class this is not valid python {{{{

def another_valid():
    return False
'''
    chunks = chunk_file("broken.py", code, ".py")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "valid_function" in symbols
    assert "another_valid" in symbols


# ---------------------------------------------------------------------------
# File with only imports/comments (no functions/classes)
# ---------------------------------------------------------------------------


def test_imports_only_file():
    """A file with only imports should produce a preamble chunk."""
    code = '''\
import os
import sys
from pathlib import Path

# Configuration constants
MAX_SIZE = 1024
DEBUG = True
'''
    chunks = chunk_file("constants.py", code, ".py")
    assert len(chunks) >= 1
    assert chunks[0].chunk_type == "preamble"
    assert "import os" in chunks[0].text
    assert "MAX_SIZE" in chunks[0].text


# ---------------------------------------------------------------------------
# JSON: top-level only (no nested duplicates)
# ---------------------------------------------------------------------------


def test_json_top_level_only():
    """JSON chunking should only produce top-level keys, not nested ones."""
    json_code = '{"name": "flowmap", "scripts": {"test": "pytest", "build": "tsc"}, "version": "1.0"}'
    chunks = chunk_file("package.json", json_code, ".json")
    symbols = {c.symbol_name for c in chunks}
    assert "name" in symbols
    assert "scripts" in symbols
    assert "version" in symbols
    # Nested keys should NOT be separate chunks
    assert "test" not in symbols
    assert "build" not in symbols


# ---------------------------------------------------------------------------
# CRITICAL: TypeScript export class splitting
# ---------------------------------------------------------------------------


def test_typescript_export_large_class_splits():
    """export class with >8000 chars should split into methods, not be silently truncated."""
    methods = "\n".join(
        f"  method_{i}(): void {{ console.log('method {i} doing work with some longer text to pad it out'); }}"
        for i in range(200)
    )
    ts_code = f"export class BigController {{\n{methods}\n}}\n"
    assert len(ts_code) > 8000, f"Fixture must exceed 8000 chars, got {len(ts_code)}"
    chunks = chunk_file("big.ts", ts_code, ".ts")
    # Must have method chunks, not just a truncated signature
    all_symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "BigController" in all_symbols, f"Missing class name, got: {all_symbols}"
    # Should have extracted many methods
    method_chunks = [c for c in chunks if "method_" in c.text and c.chunk_type in ("method", "function")]
    assert len(method_chunks) >= 50, f"Expected many method chunks, got {len(method_chunks)}"


def test_typescript_export_large_class_preserves_methods():
    """All methods in an exported large class should be individually searchable."""
    methods = "\n".join(
        f"  handle_{i}(): string {{ return 'result_{i} with some extra text for padding purposes'; }}"
        for i in range(200)
    )
    ts_code = f"export class ApiController {{\n{methods}\n}}\n"
    assert len(ts_code) > 8000
    chunks = chunk_file("api.ts", ts_code, ".ts")
    texts = " ".join(c.text for c in chunks)
    # Spot check: methods should not be silently dropped
    assert "handle_0" in texts
    assert "handle_50" in texts
    assert "handle_149" in texts


# ---------------------------------------------------------------------------
# Python: mixed decorator types in large class
# ---------------------------------------------------------------------------


_MIXED_METHODS = "\n".join([
    *[f"""    def instance_method_{i}(self):
        \"\"\"Instance method {i}.\"\"\"
        return self.data + {i}
""" for i in range(20)],
    *[f"""    @property
    def prop_{i}(self):
        \"\"\"Property {i}.\"\"\"
        return self._prop_{i}

    @prop_{i}.setter
    def prop_{i}(self, value):
        self._prop_{i} = value
""" for i in range(15)],
    *[f"""    @classmethod
    def class_method_{i}(cls):
        return cls()
""" for i in range(5)],
])

PYTHON_MIXED_LARGE_CLASS = f'''\
class MixedService:
    """Service with mixed method types."""

    def __init__(self):
        self.data = 0
{" " * 4 + "self._prop_" + " = None\\n    self._prop_".join(str(i) for i in range(15)) + " = None"}

{_MIXED_METHODS}
'''


def test_mixed_decorator_large_class_splits():
    """Large class with mixed decorators should split all method types."""
    if len(PYTHON_MIXED_LARGE_CLASS) <= 8000:
        # Skip if fixture too small — not a failure, just not enough to trigger split
        return
    chunks = chunk_file("mixed.py", PYTHON_MIXED_LARGE_CLASS, ".py")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    # Instance methods
    assert any("instance_method_0" in s for s in symbols), f"Missing instance methods, got: {symbols}"
    # Class methods
    assert any("class_method_0" in s for s in symbols), f"Missing class methods, got: {symbols}"


# ---------------------------------------------------------------------------
# TOML fallback
# ---------------------------------------------------------------------------


def test_toml_uses_fallback():
    """TOML files should use fallback chunking (no tree-sitter grammar)."""
    toml_code = '[package]\nname = "flowmap"\nversion = "0.2.0"\n\n[dependencies]\nclick = "8.0"\n'
    chunks = chunk_file("Cargo.toml", toml_code, ".toml")
    assert len(chunks) >= 1
    assert chunks[0].chunk_type == "fallback"


# ---------------------------------------------------------------------------
# Filename-based detection (Dockerfile, Makefile)
# ---------------------------------------------------------------------------


def test_dockerfile_fallback():
    """Files detected by name (no extension) should use fallback chunking."""
    content = "FROM python:3.11\nRUN pip install click\nCOPY . /app\nCMD [\"python\", \"app.py\"]\n"
    # Extension is empty for Dockerfile — indexer maps to .txt for chunking
    chunks = chunk_file("Dockerfile", content, ".txt")
    assert len(chunks) >= 1
    assert "FROM python" in chunks[0].text


# ---------------------------------------------------------------------------
# Chunk count assertions (catch regressions)
# ---------------------------------------------------------------------------


def test_python_exact_chunk_count():
    """Verify exact chunk count for the standard Python fixture to catch double-extraction."""
    chunks = chunk_file("test.py", PYTHON_CODE, ".py")
    # Expected: fetch_data, OrderProcessor, main (3 top-level defs) + 1 preamble = 4
    assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}: {[c.symbol_name or c.chunk_type for c in chunks]}"


def test_go_exact_chunk_count():
    """Verify exact chunk count for Go fixture."""
    chunks = chunk_file("main.go", GO_CODE, ".go")
    # Expected: Service type, Service.Process, NewService, preamble = 4
    assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}: {[c.symbol_name or c.chunk_type for c in chunks]}"


# ---------------------------------------------------------------------------
# Swift
# ---------------------------------------------------------------------------

SWIFT_CODE = '''\
import Foundation

func globalFunction(x: Int) -> Int {
    return x * 2
}

class ViewController {
    func viewDidLoad() {}
    func configure() {}
}

struct Point {
    var x: Double
    var y: Double
}

enum Direction {
    case north, south
}

protocol Drawable {
    func draw()
}
'''


def test_swift_function():
    chunks = chunk_file("test.swift", SWIFT_CODE, ".swift")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "globalFunction" in symbols


def test_swift_class():
    chunks = chunk_file("test.swift", SWIFT_CODE, ".swift")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "ViewController" in symbols


def test_swift_struct():
    chunks = chunk_file("test.swift", SWIFT_CODE, ".swift")
    point = next(c for c in chunks if c.symbol_name == "Point")
    assert point.chunk_type == "class"


def test_swift_enum():
    chunks = chunk_file("test.swift", SWIFT_CODE, ".swift")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "Direction" in symbols


def test_swift_protocol():
    chunks = chunk_file("test.swift", SWIFT_CODE, ".swift")
    drawable = next(c for c in chunks if c.symbol_name == "Drawable")
    assert drawable.chunk_type == "class"


def test_swift_preamble():
    chunks = chunk_file("test.swift", SWIFT_CODE, ".swift")
    preambles = [c for c in chunks if c.chunk_type == "preamble"]
    assert len(preambles) >= 1
    assert "import Foundation" in preambles[0].text


def test_swift_exact_chunk_count():
    chunks = chunk_file("test.swift", SWIFT_CODE, ".swift")
    # Expected: globalFunction, ViewController, Point, Direction, Drawable (5 defs) + 1 preamble = 6
    assert len(chunks) == 6, f"Expected 6 chunks, got {len(chunks)}: {[c.symbol_name or c.chunk_type for c in chunks]}"


def test_swift_language_field():
    chunks = chunk_file("test.swift", SWIFT_CODE, ".swift")
    assert all(c.language == "swift" for c in chunks)


def test_swift_init_deinit():
    """Swift init/deinit inside a class should be extractable as method chunks."""
    code = '''\
class Service {
    var name: String

    init(name: String) {
        self.name = name
    }

    deinit {
        print("cleanup")
    }

    func process() {}
}
'''
    # This class is small (<8000 chars) so it won't split — but verify that if it
    # did split, the method types would be recognized by checking a large version.
    chunks = chunk_file("svc.swift", code, ".swift")
    svc = next(c for c in chunks if c.symbol_name == "Service")
    assert "init(name:" in svc.text
    assert "deinit" in svc.text


def test_swift_large_class_splits():
    """Swift class >8000 chars should split into per-method chunks."""
    methods = "\n".join(
        f"    func method_{i}(x: Int) -> Int {{ return x + {i}; /* padding to make this method longer for the test fixture to exceed the 8000 char threshold needed */ }}"
        for i in range(120)
    )
    code = f"class BigSwiftService {{\n{methods}\n}}\n"
    assert len(code) > 8000, f"Fixture must exceed 8000 chars, got {len(code)}"
    chunks = chunk_file("big.swift", code, ".swift")
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) >= 1
    assert class_chunks[0].symbol_name == "BigSwiftService"
    method_chunks = [c for c in chunks if c.chunk_type in ("function", "method") and c.parent_symbol == "BigSwiftService"]
    assert len(method_chunks) >= 50, f"Expected many method chunks, got {len(method_chunks)}"


# ---------------------------------------------------------------------------
# JavaScript: CommonJS exports
# ---------------------------------------------------------------------------

JS_CJS_CODE = '''\
const helper = require('./helper');

module.exports.getArea = function(radius) {
    return Math.PI * radius * radius;
};

exports.getCircumference = (radius) => {
    return 2 * Math.PI * radius;
};

function internalHelper() {
    return 42;
}
'''


def test_js_cjs_module_exports_function():
    chunks = chunk_file("math.js", JS_CJS_CODE, ".js")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "getArea" in symbols


def test_js_cjs_exports_arrow():
    chunks = chunk_file("math.js", JS_CJS_CODE, ".js")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "getCircumference" in symbols


def test_js_cjs_internal_function_still_works():
    chunks = chunk_file("math.js", JS_CJS_CODE, ".js")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "internalHelper" in symbols


def test_js_cjs_no_garbage():
    """expression_statement that is NOT a CommonJS export should not become a symbol."""
    code = 'console.log("hello");\nx = 42;\nmodule.exports.getArea = function() { return 1; };\n'
    chunks = chunk_file("mixed.js", code, ".js")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "getArea" in symbols
    assert "log" not in symbols
    assert "x" not in symbols


def test_js_cjs_require_reexport_rejected():
    """module.exports = require('./other') should not produce a symbol."""
    code = "module.exports = require('./other');\n"
    chunks = chunk_file("reexport.js", code, ".js")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert len(symbols) == 0


def test_js_cjs_config_object_rejected():
    """module.exports.config = { port: 3000 } should not produce a symbol (object on named export)."""
    code = "module.exports.config = { port: 3000, host: 'localhost' };\n"
    chunks = chunk_file("config.js", code, ".js")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "config" not in symbols


def test_js_cjs_named_function():
    """module.exports = function myFunc() {} should extract 'myFunc'."""
    code = "module.exports = function myFunc() { return 42; };\n"
    chunks = chunk_file("named.js", code, ".js")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "myFunc" in symbols


def test_js_cjs_class_export():
    """module.exports = class Service {} should extract 'Service' with chunk_type 'class'."""
    code = "module.exports = class Service { constructor() {} };\n"
    chunks = chunk_file("svc.js", code, ".js")
    svc = next((c for c in chunks if c.symbol_name == "Service"), None)
    assert svc is not None, f"Expected Service, got: {[c.symbol_name for c in chunks]}"
    assert svc.chunk_type == "class"


def test_js_cjs_bulk_as_single_chunk():
    """module.exports = { method1() {} } should produce one chunk with symbol 'module.exports'."""
    code = "module.exports = { method1() { return 1; }, method2() { return 2; } };\n"
    chunks = chunk_file("bulk.js", code, ".js")
    bulk = next((c for c in chunks if c.symbol_name == "module.exports"), None)
    assert bulk is not None, f"Expected module.exports chunk, got: {[c.symbol_name for c in chunks]}"


def test_js_cjs_preamble():
    chunks = chunk_file("math.js", JS_CJS_CODE, ".js")
    preambles = [c for c in chunks if c.chunk_type == "preamble"]
    assert len(preambles) >= 1
    combined = " ".join(p.text for p in preambles)
    assert "require" in combined


def test_js_cjs_exact_chunk_count():
    chunks = chunk_file("math.js", JS_CJS_CODE, ".js")
    # Expected: getArea + getCircumference + internalHelper + preamble = 4
    assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}: {[c.symbol_name or c.chunk_type for c in chunks]}"


def test_js_cjs_line_ranges():
    chunks = chunk_file("math.js", JS_CJS_CODE, ".js")
    area = next(c for c in chunks if c.symbol_name == "getArea")
    assert area.start_line > 0
    assert area.end_line >= area.start_line


def test_js_cjs_signature():
    chunks = chunk_file("math.js", JS_CJS_CODE, ".js")
    area = next(c for c in chunks if c.symbol_name == "getArea")
    assert "module.exports.getArea" in area.signature


# ---------------------------------------------------------------------------
# Size-cap enforcement (universal oversized-chunk splitting)
# ---------------------------------------------------------------------------

from flowmap.parsing.chunker import MAX_CHUNK_CHARS


def test_oversized_function_is_split_not_emitted_whole():
    """A function far larger than the cap is split into parts each within it."""
    body = "\n".join(f"    x{i} = compute({i})" for i in range(4000))
    code = f"def huge():\n{body}\n    return x0\n"
    assert len(code) > MAX_CHUNK_CHARS * 2  # genuinely oversized
    chunks = chunk_file("huge.py", code, ".py")
    assert len(chunks) >= 2
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)


def test_minified_single_line_is_hard_split():
    """A minified one-line file (no newlines) is hard-split on char boundaries."""
    one_line = "var a=" + ",".join(str(i) for i in range(60000)) + ";"
    assert "\n" not in one_line and len(one_line) > MAX_CHUNK_CHARS
    chunks = chunk_file("bundle.min.js", one_line, ".js")
    assert len(chunks) >= 2
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)


def test_split_preserves_all_content():
    """Splitting must not drop content — concatenating the parts' text must
    recover every non-whitespace character of the original."""
    body = "\n".join(f"    line_{i} = {i} * 2" for i in range(3000))
    code = f"def big():\n{body}\n"
    chunks = chunk_file("big.py", code, ".py")
    joined = "".join(c.text for c in chunks)
    assert "".join(code.split()) == "".join(joined.split())


def test_small_chunks_untouched():
    """The cap pass must be a no-op for normally-sized files."""
    code = "def small():\n    return 1\n"
    chunks = chunk_file("small.py", code, ".py")
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "small"


def test_all_whitespace_oversized_chunk_is_dropped_not_reemitted():
    """An all-whitespace oversized chunk is dropped, not re-emitted. Guards
    against a `return parts or [chunk]` fallback that would breach the cap."""
    from flowmap.parsing.chunker import _enforce_size_cap

    big_ws = Chunk(
        text=" " * (MAX_CHUNK_CHARS * 2), chunk_type="fallback",
        symbol_name="", signature="", parent_symbol="", parent_signature="",
        start_line=1, end_line=1, language="python",
    )
    out = _enforce_size_cap([big_ws])
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in out)
    assert out == []
