"""Tests for query classification — determines how search sources are weighted."""

from flowmap.search.hybrid import classify_query


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

def test_camel_case():
    assert classify_query("processOrder") == "identifier"

def test_snake_case():
    assert classify_query("process_order") == "identifier"

def test_pascal_case():
    assert classify_query("AuthMiddleware") == "identifier"

def test_dotted_path():
    assert classify_query("os.path.join") == "identifier"

def test_class_method():
    assert classify_query("VectorStore.search_vector") == "identifier"


# ---------------------------------------------------------------------------
# Natural language
# ---------------------------------------------------------------------------

def test_single_lowercase_word():
    assert classify_query("retry") == "natural_language"

def test_simple_question():
    assert classify_query("how does auth work") == "natural_language"

def test_natural_sentence():
    assert classify_query("where is the retry logic") == "natural_language"

def test_single_common_word():
    assert classify_query("authentication") == "natural_language"


# ---------------------------------------------------------------------------
# Mixed
# ---------------------------------------------------------------------------

def test_nl_with_identifier():
    assert classify_query("what is AuthMiddleware") == "mixed"

def test_nl_with_snake_case():
    assert classify_query("find process_order function") == "mixed"

def test_nl_with_camel_case():
    assert classify_query("how does processOrder work") == "mixed"

def test_nl_with_dotted():
    assert classify_query("where is os.path used") == "mixed"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_uppercase_word():
    """GET, HTTP, API etc. — all-caps tokens are identifiers (constants, HTTP methods)."""
    assert classify_query("GET") == "identifier"
    assert classify_query("HTTP") == "identifier"
    assert classify_query("API") == "identifier"

def test_single_char():
    assert classify_query("x") == "natural_language"

def test_number():
    assert classify_query("404") == "natural_language"

def test_empty_like():
    assert classify_query("a") == "natural_language"
