"""Tests for Rust AST chunking — functions, structs, enums, traits, impls."""

from flowmap.parsing import chunk_file


RUST_FREE_FN_CODE = '''\
fn add(a: i32, b: i32) -> i32 {
    a + b
}
'''


def test_rust_free_function():
    chunks = chunk_file("lib.rs", RUST_FREE_FN_CODE, ".rs")
    functions = [c for c in chunks if c.chunk_type == "function"]
    assert any(c.symbol_name == "add" for c in functions)


RUST_STRUCT_CODE = '''\
struct Point {
    x: f64,
    y: f64,
}
'''


def test_rust_struct():
    chunks = chunk_file("point.rs", RUST_STRUCT_CODE, ".rs")
    classes = [c for c in chunks if c.chunk_type == "class"]
    assert any(c.symbol_name == "Point" for c in classes)


RUST_ENUM_CODE = '''\
enum Color {
    Red,
    Green,
    Blue,
}
'''


def test_rust_enum():
    chunks = chunk_file("color.rs", RUST_ENUM_CODE, ".rs")
    classes = [c for c in chunks if c.chunk_type == "class"]
    assert any(c.symbol_name == "Color" for c in classes)


RUST_TRAIT_CODE = '''\
trait Greet {
    fn hello(&self);
}
'''


def test_rust_trait():
    chunks = chunk_file("greet.rs", RUST_TRAIT_CODE, ".rs")
    classes = [c for c in chunks if c.chunk_type == "class"]
    assert any(c.symbol_name == "Greet" for c in classes)


RUST_INHERENT_IMPL_CODE = '''\
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}
'''


def test_rust_impl_method_qualified_name():
    """A method inside `impl Point` should be `Point.new` with chunk_type=method."""
    chunks = chunk_file("point.rs", RUST_INHERENT_IMPL_CODE, ".rs")
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert any(c.symbol_name == "Point.new" for c in methods), \
        f"expected Point.new in {[c.symbol_name for c in chunks]}"


RUST_TRAIT_IMPL_CODE = '''\
trait Greet {
    fn hello(&self);
}

struct Person;

impl Greet for Person {
    fn hello(&self) {
        println!("hi");
    }
}
'''


def test_rust_trait_impl_qualified_name():
    """A method in `impl Greet for Person` should be `Greet.hello`, parent=Greet."""
    chunks = chunk_file("greet.rs", RUST_TRAIT_IMPL_CODE, ".rs")
    methods = [c for c in chunks if c.chunk_type == "method"]
    method = next((m for m in methods if m.symbol_name == "Greet.hello"), None)
    assert method is not None, f"expected Greet.hello in {[c.symbol_name for c in chunks]}"
    assert method.parent_symbol == "Greet"


RUST_TYPE_CONST_CODE = '''\
type Result<T> = std::result::Result<T, Error>;

const MAX_RETRIES: u32 = 3;

static NAME: &str = "flowmap";
'''


def test_rust_type_alias_and_consts():
    chunks = chunk_file("aliases.rs", RUST_TYPE_CONST_CODE, ".rs")
    symbols = {c.symbol_name: c.chunk_type for c in chunks if c.symbol_name}
    assert symbols.get("Result") == "class", f"got {symbols}"
    assert symbols.get("MAX_RETRIES") == "property", f"got {symbols}"
    assert symbols.get("NAME") == "property", f"got {symbols}"


def test_rust_language_field():
    chunks = chunk_file("lib.rs", RUST_FREE_FN_CODE, ".rs")
    assert all(c.language == "rust" for c in chunks)


def test_rust_signature_extraction():
    chunks = chunk_file("lib.rs", RUST_FREE_FN_CODE, ".rs")
    fn = next(c for c in chunks if c.symbol_name == "add")
    assert fn.signature.startswith("fn add")
    assert "->" in fn.signature


def test_rust_line_ranges():
    chunks = chunk_file("lib.rs", RUST_FREE_FN_CODE, ".rs")
    fn = next(c for c in chunks if c.symbol_name == "add")
    assert fn.start_line == 1
    assert fn.end_line == 3


def test_rust_impl_does_not_emit_itself():
    """The impl block must not appear as a chunk (only its methods)."""
    chunks = chunk_file("point.rs", RUST_INHERENT_IMPL_CODE, ".rs")
    assert "impl Point" not in {c.symbol_name for c in chunks}
    assert all("impl " not in c.symbol_name for c in chunks)


RUST_REALISTIC_CODE = '''\
use std::fmt;

#[derive(Debug)]
pub struct Order {
    id: u64,
    total: u64,
}

pub enum OrderError {
    NotFound,
    AlreadyPaid,
}

pub trait Repository {
    fn find(&self, id: u64) -> Result<Order, OrderError>;
    fn save(&self, order: &Order) -> Result<(), OrderError>;
}

impl Order {
    pub fn new(id: u64, total: u64) -> Self {
        Order { id, total }
    }

    pub fn is_paid(&self) -> bool {
        self.total > 0
    }
}

impl fmt::Display for OrderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}
'''


def test_rust_realistic_mixed_file():
    """A realistic Rust file with struct, enum, trait, inherent impl, trait impl
    must produce all expected symbols and no junk."""
    chunks = chunk_file("order.rs", RUST_REALISTIC_CODE, ".rs")
    symbols = {c.symbol_name: c.chunk_type for c in chunks if c.symbol_name}

    # Top-level items
    assert symbols.get("Order") == "class", f"missing Order: {symbols}"
    assert symbols.get("OrderError") == "class", f"missing OrderError: {symbols}"
    assert symbols.get("Repository") == "class", f"missing Repository: {symbols}"

    # Inherent impl methods
    assert symbols.get("Order.new") == "method", f"missing Order.new: {symbols}"
    assert symbols.get("Order.is_paid") == "method", f"missing Order.is_paid: {symbols}"

    # Trait impl method — qualified by trait's base identifier (Display, not fmt::Display)
    assert symbols.get("Display.fmt") == "method", f"missing Display.fmt: {symbols}"


RUST_GENERIC_IMPL_CODE = '''\
struct Container<T> {
    item: T,
}

impl<T> Container<T> {
    fn new(item: T) -> Self {
        Container { item }
    }
}

impl<T: Send> Container<T> where T: Clone {
    fn clone_item(&self) -> T {
        self.item.clone()
    }
}
'''


def test_rust_generic_impl_uses_base_identifier():
    """`impl<T> Container<T>` must produce `Container.new`, not `Container<T>.new`."""
    chunks = chunk_file("container.rs", RUST_GENERIC_IMPL_CODE, ".rs")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "Container.new" in symbols, f"expected Container.new, got {symbols}"
    assert "Container.clone_item" in symbols, f"expected Container.clone_item, got {symbols}"
    # Negative — no `<T>` in the symbol
    assert not any("<" in s for s in symbols), f"unexpected generic in symbol: {symbols}"


RUST_ASSOCIATED_ITEMS_CODE = '''\
struct Config {
    debug: bool,
}

impl Config {
    const MAX: u32 = 100;
    static COUNTER: u32 = 0;
    type Item = u64;

    fn new() -> Self {
        Config { debug: false }
    }
}
'''


def test_rust_impl_associated_items_extracted():
    """const, static, and type items inside an impl block must be extracted
    as their own chunks, not buried in the preamble."""
    chunks = chunk_file("config.rs", RUST_ASSOCIATED_ITEMS_CODE, ".rs")
    symbols = {c.symbol_name: c.chunk_type for c in chunks if c.symbol_name}
    assert symbols.get("Config.MAX") == "property", f"missing Config.MAX: {symbols}"
    assert symbols.get("Config.COUNTER") == "property", f"missing Config.COUNTER: {symbols}"
    assert symbols.get("Config.Item") == "class", f"missing Config.Item: {symbols}"
    assert symbols.get("Config.new") == "method", f"missing Config.new: {symbols}"


RUST_TRAIT_DEFAULT_METHODS_CODE = '''\
trait WithDefaults {
    fn must_implement(&self) -> i32;

    fn default_impl(&self) -> i32 {
        42
    }
}
'''


def test_rust_trait_default_method_bodies_extracted():
    """Default method bodies inside a trait must be extracted as method chunks,
    not hidden inside the trait declaration. Default bodies use chunk_type
    'method_default' so they don't collide with impl overrides."""
    chunks = chunk_file("defaults.rs", RUST_TRAIT_DEFAULT_METHODS_CODE, ".rs")
    by_name = {c.symbol_name: c.chunk_type for c in chunks if c.chunk_type in ("method", "method_default")}
    assert "WithDefaults.default_impl" in by_name, f"missing default method: {by_name}"
    # The required (no body) declaration should also be visible as a method
    assert "WithDefaults.must_implement" in by_name, f"missing required method: {by_name}"
    # The default method body should be tagged distinctly from a regular method
    assert by_name["WithDefaults.default_impl"] == "method_default"


RUST_DOC_AND_ATTRS_CODE = '''\
/// Handles authentication for the user.
#[derive(Debug, Clone)]
pub struct AuthService {
    secret: String,
}

/// Greets the caller.
pub fn hello() -> String {
    String::from("hi")
}
'''


def test_rust_doc_comments_and_attributes_in_chunk_text():
    """`///` doc comments and `#[...]` attributes preceding an item must be
    included in the chunk's text — not stranded in the preamble."""
    chunks = chunk_file("auth.rs", RUST_DOC_AND_ATTRS_CODE, ".rs")
    struct_chunk = next(c for c in chunks if c.symbol_name == "AuthService")
    assert "/// Handles authentication" in struct_chunk.text, \
        f"missing doc comment in struct chunk: {struct_chunk.text[:120]!r}"
    assert "#[derive(Debug, Clone)]" in struct_chunk.text, \
        f"missing attribute in struct chunk: {struct_chunk.text[:120]!r}"

    fn_chunk = next(c for c in chunks if c.symbol_name == "hello")
    assert "/// Greets the caller." in fn_chunk.text, \
        f"missing doc comment in fn chunk: {fn_chunk.text[:120]!r}"


def test_rust_blank_line_breaks_trivia_attachment():
    """A doc comment separated from the item by a blank line must NOT be attached."""
    code = '''\
/// Orphan doc comment

fn foo() {}
'''
    chunks = chunk_file("orphan.rs", code, ".rs")
    fn_chunk = next(c for c in chunks if c.symbol_name == "foo")
    assert "/// Orphan" not in fn_chunk.text, \
        "doc comment separated by blank line should not attach"


RUST_MOD_CODE = '''\
mod outer {
    pub fn inner_fn() -> i32 {
        42
    }

    mod nested {
        pub fn deeply_nested() {}
    }
}
'''


def test_rust_mod_recursion_extracts_nested_items():
    """Items inside a `mod` block must be extracted, not silently dropped.

    Nested mod functions get qualified names: `outer.inner_fn`, `outer.nested.deeply_nested`.
    """
    chunks = chunk_file("mods.rs", RUST_MOD_CODE, ".rs")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "outer.inner_fn" in symbols, f"missing outer.inner_fn: {symbols}"
    assert "outer.nested.deeply_nested" in symbols, \
        f"missing outer.nested.deeply_nested: {symbols}"


def test_rust_mod_recursion_preserves_impl_methods():
    """Impl methods inside a mod must still be extracted with qualified names."""
    code = '''\
mod api {
    pub struct Service;

    impl Service {
        pub fn new() -> Self { Service }
    }
}
'''
    chunks = chunk_file("service.rs", code, ".rs")
    methods = {c.symbol_name for c in chunks if c.chunk_type == "method"}
    assert "api.Service.new" in methods, f"missing api.Service.new: {methods}"


def test_rust_mod_recursion_extracts_top_level_test_module():
    """The very common `#[cfg(test)] mod tests { #[test] fn test_x() {} }` pattern
    must extract the test function."""
    code = '''\
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thing() {
        assert_eq!(2 + 2, 4);
    }
}
'''
    chunks = chunk_file("with_tests.rs", code, ".rs")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "tests.test_thing" in symbols, f"missing tests.test_thing: {symbols}"


RUST_IMPL_METHOD_TRIVIA_CODE = '''\
struct S;

impl S {
    /// Constructs a new S.
    #[inline]
    pub fn new() -> Self {
        S
    }
}
'''


def test_rust_impl_method_gets_doc_comment_and_attribute():
    """Doc comments and attributes above an impl method must be in the
    method's chunk text, not stranded in the preamble."""
    chunks = chunk_file("x.rs", RUST_IMPL_METHOD_TRIVIA_CODE, ".rs")
    method = next(c for c in chunks if c.symbol_name == "S.new")
    assert "/// Constructs a new S." in method.text, \
        f"missing doc comment in impl method: {method.text[:120]!r}"
    assert "#[inline]" in method.text, \
        f"missing attribute in impl method: {method.text[:120]!r}"


RUST_TRAIT_METHOD_TRIVIA_CODE = '''\
trait Greet {
    /// Required behavior for the greeter.
    fn hello(&self);
}
'''


def test_rust_trait_method_gets_doc_comment():
    """Doc comments above a trait method (required, no body) must attach."""
    chunks = chunk_file("t.rs", RUST_TRAIT_METHOD_TRIVIA_CODE, ".rs")
    method = next(c for c in chunks if c.symbol_name == "Greet.hello")
    assert "/// Required behavior" in method.text, \
        f"missing doc comment in trait method: {method.text[:120]!r}"


def test_rust_mod_does_not_leak_braces_into_preamble():
    """A mod block's outer braces must not appear in the preamble as stray
    text — `_extract_preamble` assumes non-overlapping byte ranges, and
    the mod's full range was overlapping its children's ranges."""
    code = '''\
mod m {
    pub fn inner() -> i32 {
        42
    }
}
'''
    chunks = chunk_file("x.rs", code, ".rs")
    preambles = [c for c in chunks if c.chunk_type == "preamble"]
    if preambles:
        preamble_text = "\n".join(c.text for c in preambles)
        # The closing brace of `mod m` should be inside an extracted range, not
        # leaked into preamble.
        assert "}" not in preamble_text, \
            f"preamble leaked mod closing brace: {preamble_text!r}"


def test_rust_nested_mod_no_brace_leak():
    """Deeper nesting also doesn't leak braces into preamble."""
    code = '''\
mod outer {
    mod inner {
        pub fn deep() {}
    }
}
'''
    chunks = chunk_file("x.rs", code, ".rs")
    preambles = [c for c in chunks if c.chunk_type == "preamble"]
    if preambles:
        preamble_text = "\n".join(c.text for c in preambles)
        assert "}" not in preamble_text, \
            f"preamble leaked braces: {preamble_text!r}"


def test_rust_trait_class_chunk_is_signature_only():
    """When a trait has per-method chunks, the trait's class chunk should
    contain only the signature line — not duplicate the method bodies."""
    code = '''\
trait Greeter {
    /// Says hi.
    fn hello(&self) -> String;
}
'''
    chunks = chunk_file("t.rs", code, ".rs")
    class_chunks = [c for c in chunks if c.chunk_type == "class" and c.symbol_name == "Greeter"]
    assert len(class_chunks) == 1
    trait_class = class_chunks[0]
    # The class chunk must not contain the method body / doc.
    assert "fn hello" not in trait_class.text, \
        f"trait class chunk duplicates method: {trait_class.text!r}"
    # It should contain the trait signature itself.
    assert "trait Greeter" in trait_class.text, \
        f"trait class chunk missing signature: {trait_class.text!r}"
    # Stronger assertion (regression test for N9): the chunk must include
    # both the opening `{` and the closing `}` braces, not just one.
    assert "trait Greeter {" in trait_class.text, \
        f"trait class chunk missing opening brace: {trait_class.text!r}"
    assert trait_class.text.rstrip().endswith("}"), \
        f"trait class chunk missing closing brace: {trait_class.text!r}"


def test_rust_trait_class_chunk_does_not_duplicate_trivia():
    """Regression test for N11: when a trait itself has a doc comment or
    `#[..]` attribute, the class chunk must not duplicate them. The
    signature-only body-stripping logic was previously adding trivia twice."""
    code = '''\
/// Greeter says hi to people.
#[derive(Debug)]
pub trait Greeter {
    fn hello(&self);
}
'''
    chunks = chunk_file("t.rs", code, ".rs")
    trait_class = next(c for c in chunks if c.symbol_name == "Greeter" and c.chunk_type == "class")
    # The doc comment must appear exactly once.
    assert trait_class.text.count("/// Greeter says hi to people.") == 1, \
        f"doc comment duplicated: {trait_class.text!r}"
    # The attribute must appear exactly once.
    assert trait_class.text.count("#[derive(Debug)]") == 1, \
        f"attribute duplicated: {trait_class.text!r}"


def test_rust_method_default_accepted_by_cli_choice():
    """Regression test for N12: `flowmap symbols --type method_default`
    must be a valid CLI choice so users can find trait default method bodies."""
    from click.testing import CliRunner
    from flowmap.cli import main
    runner = CliRunner()
    # We don't need the command to succeed end-to-end (no index); we just
    # need it not to reject the type choice with a "Invalid value" error.
    # Use --help on the symbols command to trigger choice validation.
    result = runner.invoke(main, ["symbols", "--help"], catch_exceptions=False)
    assert "method_default" in result.output, \
        f"`method_default` missing from --type choices: {result.output[:500]!r}"


def test_rust_stacked_attributes_preserve_newlines():
    """Multiple `#[..]` attributes stacked above an item must keep their
    newlines in the chunk text — not be concatenated as `#[a]#[b]`."""
    code = '''\
#[derive(Debug)]
#[inline]
struct S;
'''
    chunks = chunk_file("s.rs", code, ".rs")
    s = next(c for c in chunks if c.symbol_name == "S")
    assert "#[derive(Debug)]\n#[inline]" in s.text, \
        f"attributes not newline-separated: {s.text!r}"


def test_rust_mod_declaration_without_body():
    """`mod foo;` (a forward declaration of an external module) must still
    produce a chunk so users can search for the mod name."""
    code = '''\
mod lib;

fn main() {}
'''
    chunks = chunk_file("main.rs", code, ".rs")
    symbols = {c.symbol_name for c in chunks if c.symbol_name}
    assert "lib" in symbols, f"missing mod lib declaration: {symbols}"


def test_rust_negative_impl_emits_chunk():
    """`impl !Send for Foo {}` (negative impl, often empty body) must emit
    a chunk so users can search for type-system opt-outs."""
    code = '''\
struct Foo;

impl !Send for Foo {}
'''
    chunks = chunk_file("x.rs", code, ".rs")
    # We expect at least one chunk for the negative impl
    neg_chunks = [c for c in chunks if "Send" in c.symbol_name and c.chunk_type == "class"]
    assert len(neg_chunks) >= 1, f"missing negative impl chunk: {[c.symbol_name for c in chunks]}"


def test_rust_inner_doc_comment_does_not_attach_to_next_item():
    """`//!` inner doc comments document the enclosing module, not the next
    item. They must NOT be attached to the next item (semantically wrong)
    — they should remain in the preamble instead.

    Tests both with and without an intervening blank line, since blank
    lines are not a reliable distinguisher of inner vs outer docs.
    """
    # Case 1: blank line between //! and item — preamble by gap
    code1 = '''\
//! This module does X.

fn foo() {}
'''
    chunks = chunk_file("x.rs", code1, ".rs")
    fn = next(c for c in chunks if c.symbol_name == "foo")
    assert "//! This module does X" not in fn.text

    # Case 2: NO blank line — must still NOT attach (semantic, not positional)
    code2 = '''\
//! This module does X.
/// Outer doc for foo.
fn foo() {}
'''
    chunks = chunk_file("x.rs", code2, ".rs")
    fn = next(c for c in chunks if c.symbol_name == "foo")
    assert "//! This module does X" not in fn.text, \
        f"inner doc comment incorrectly attached: {fn.text[:120]!r}"
    # Outer doc should still attach
    assert "/// Outer doc for foo." in fn.text


def test_rust_macro_rules_definition_extracted():
    """`macro_rules! my_macro { ... }` must be extracted as a chunk so users
    can search macro bodies."""
    code = '''\
macro_rules! my_macro {
    ($x:expr) => { $x + 1 };
}
'''
    chunks = chunk_file("m.rs", code, ".rs")
    symbols = {c.symbol_name: c.chunk_type for c in chunks if c.symbol_name}
    assert symbols.get("my_macro") == "function", f"missing my_macro: {symbols}"


RUST_TRAIT_DEFAULT_AND_OVERRIDE_CODE = '''\
trait Greet {
    fn hello(&self) -> String {
        String::from("hi")
    }
}

struct Person;

impl Greet for Person {
    fn hello(&self) -> String {
        format!("hello, {}!", "world")
    }
}
'''


def test_rust_trait_default_and_impl_override_dont_collide():
    """When a trait has a default method body AND an impl overrides it,
    the two chunks must not collide on the same symbol — they should be
    distinguishable so search results don't mix them up."""
    chunks = chunk_file("x.rs", RUST_TRAIT_DEFAULT_AND_OVERRIDE_CODE, ".rs")
    greet_hello = [c for c in chunks if c.symbol_name == "Greet.hello"]
    # Two chunks: the trait default body and the impl override.
    assert len(greet_hello) == 2, \
        f"expected 2 Greet.hello chunks, got {len(greet_hello)}: {[(c.chunk_type, c.symbol_name) for c in chunks]}"
    # They must have different chunk_types so search can distinguish them.
    chunk_types = {c.chunk_type for c in greet_hello}
    assert len(chunk_types) == 2, \
        f"expected different chunk_types, got {chunk_types}"
