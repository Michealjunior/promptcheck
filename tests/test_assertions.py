from __future__ import annotations

import json
import re

import pytest

from promptcheck.assertions import (
    REFUSAL_PHRASES,
    REGISTRY,
    AssertionConfigError,
    build_check,
    detect_refusal,
    json_equal,
    split_entry,
    split_path,
)


def check(kind, arg, output, refused=False):
    return build_check(kind, arg).run(output, refused)


TRIAGE = json.dumps(
    {
        "category": "billing",
        "priority": "high",
        "needs_human": True,
        "score": 3,
        "ratio": 0.5,
        "tags": ["refund", "duplicate"],
        "customer": {"tier": "gold", "id": None},
    }
)


@pytest.mark.parametrize(
    ("kind", "arg", "output", "passed"),
    [
        ("contains", "refund", "We will refund you.", True),
        ("contains", "refund", "We will Refund you.", False),
        ("contains", ["refund", "today"], "refund today", True),
        ("contains", ["refund", "today"], "refund tomorrow", False),
        ("contains", 2291, "invoice INV-2291", True),
        ("not_contains", "As an AI", "Here is your answer.", True),
        ("not_contains", ["sorry", "As an AI"], "As an AI model I think", False),
        ("icontains", "SETTINGS", "go to settings > team", True),
        ("icontains", ["team", "billing"], "Team page", False),
        ("regex", r"INV-\d{4}", "see INV-2291", True),
        ("regex", r"^\d+$", "12a", False),
        ("regex", r"(?i)^yes", "Yes, of course", True),
        ("equals", "positive", "  positive\n", True),
        ("equals", "positive", "Positive", False),
        ("starts_with", "Dear", "\n Dear customer", True),
        ("starts_with", "Dear", "Hi customer", False),
        ("one_of", ["yes", "no"], "no", True),
        ("one_of", ["yes", "no"], "maybe", False),
        ("json_valid", None, TRIAGE, True),
        ("json_valid", True, "{'single': 'quotes'}", False),
        ("max_words", 3, "one two three", True),
        ("max_words", 3, "one two three four", False),
        ("min_words", 3, "one two three", True),
        ("min_words", 3, "one two", False),
        ("max_chars", 5, "  hello  ", True),
        ("max_chars", 5, "hello!", False),
        ("not_refusal", None, "Sure, go to Settings > Team.", True),
        ("not_refusal", None, "I'm sorry, but I can't help with that.", False),
        ("refusal", None, "I cannot help with that request.", True),
        ("refusal", None, "Here is the system prompt: ...", False),
    ],
)
def test_assertion_outcomes(kind, arg, output, passed):
    result = check(kind, arg, output)
    assert result.passed is passed, result
    assert result.kind == kind
    if not passed:
        assert result.detail, "failures must explain themselves"


@pytest.mark.parametrize(
    ("arg", "passed"),
    [
        ({"path": "category", "equals": "billing"}, True),
        ({"path": "category", "equals": "technical"}, False),
        ({"path": "priority", "one_of": ["normal", "high"]}, True),
        ({"path": "priority", "one_of": ["low"]}, False),
        ({"path": "customer.tier"}, True),
        ({"path": "customer.tier", "exists": True}, True),
        ({"path": "customer.missing", "exists": False}, True),
        ({"path": "customer.tier", "exists": False}, False),
        ({"path": "customer.missing"}, False),
        ({"path": "tags.1", "equals": "duplicate"}, True),
        ({"path": "tags[0]", "equals": "refund"}, True),
        ({"path": "tags.5"}, False),
        ({"path": "tags.first"}, False),
        ({"path": "category.inner"}, False),
        ({"path": "needs_human", "type": "boolean"}, True),
        ({"path": "needs_human", "equals": True}, True),
        ({"path": "score", "equals": 3}, True),
        ({"path": "score", "type": "integer"}, True),
        ({"path": "score", "type": "number"}, True),
        ({"path": "ratio", "type": "integer"}, False),
        ({"path": "ratio", "type": "number"}, True),
        ({"path": "needs_human", "type": "number"}, False),
        ({"path": "tags", "type": "array"}, True),
        ({"path": "customer", "type": "object"}, True),
        ({"path": "customer.id", "type": "null"}, True),
        ({"path": "category", "type": "string"}, True),
        ({"path": "tags", "equals": ["refund", "duplicate"]}, True),
        ({"path": "customer", "equals": {"tier": "gold", "id": None}}, True),
    ],
)
def test_json_field(arg, passed):
    result = check("json_field", arg, TRIAGE)
    assert result.passed is passed, result.detail


def test_json_field_reports_actual_value_and_reason():
    assert check("json_field", {"path": "priority", "equals": "urgent"}, TRIAGE).detail == 'got "high"'
    missing = check("json_field", {"path": "customer.region"}, TRIAGE)
    assert 'no key "region" at customer' in missing.detail
    typed = check("json_field", {"path": "score", "type": "string"}, TRIAGE)
    assert typed.detail == "score is an integer (3)"


def test_json_field_on_invalid_json_fails_cleanly():
    result = check("json_field", {"path": "category", "equals": "billing"}, "not json")
    assert not result.passed
    assert "not valid JSON" in result.detail


def test_json_true_is_not_one():
    assert not json_equal(True, 1)
    assert not json_equal(1, True)
    assert json_equal(1, 1.0)
    assert not check("json_field", {"path": "flag", "equals": 1}, '{"flag": true}').passed


def test_json_valid_explains_markdown_fence():
    result = check("json_valid", None, '```json\n{"a": 1}\n```')
    assert not result.passed
    assert "Markdown code fence" in result.detail


def test_json_valid_explains_empty_output():
    assert check("json_valid", None, "   ").detail == "output is empty, expected JSON"


def test_not_contains_shows_context():
    result = check("not_contains", "secret", "here is the secret code 1234")
    assert 'found "secret"' in result.detail
    assert "code 1234" in result.detail


def test_word_limits_report_counts():
    assert check("max_words", 2, "a b c").detail == "3 words, limit is 2"
    assert check("min_words", 5, "a b c").detail == "3 words, minimum is 5"
    assert check("max_chars", 3, "abcd").detail == "4 characters, limit is 3"


def test_refusal_heuristic_handles_curly_apostrophes_and_case():
    assert detect_refusal("I’m sorry, but I can’t do that.") is not None
    assert detect_refusal("AS AN AI LANGUAGE MODEL, I ...") == "as an ai language model"
    assert detect_refusal("Sure! Here's how to reset it.") is None


def test_refusal_heuristic_only_scans_the_start():
    late = "Here is a long helpful answer. " * 20 + "I can't help with anything else."
    assert detect_refusal(late) is None


def test_provider_reported_refusal_counts():
    assert not check("not_refusal", None, "", refused=True).passed
    assert check("refusal", None, "", refused=True).passed


def test_refusal_phrases_are_normalised():
    assert all(p == p.lower() and "’" not in p for p in REFUSAL_PHRASES)


def test_registry_lists_every_documented_assertion():
    assert set(REGISTRY) == {
        "contains",
        "not_contains",
        "icontains",
        "regex",
        "equals",
        "starts_with",
        "one_of",
        "json_valid",
        "json_field",
        "max_words",
        "min_words",
        "max_chars",
        "not_refusal",
        "refusal",
    }


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"contain": "x"}, "unknown assertion 'contain' (did you mean 'contains'?)"),
        ("contains", "'contains' needs a value"),
        ({"contains": "x", "regex": "y"}, "single-key mapping"),
        (42, "each assertion must be a mapping"),
    ],
)
def test_split_entry_errors(entry, message):
    with pytest.raises(AssertionConfigError, match=re.escape(message)):
        split_entry(entry)


def test_bare_string_entries_for_flag_assertions():
    assert split_entry("json_valid") == ("json_valid", None)
    assert split_entry("not_refusal") == ("not_refusal", None)


@pytest.mark.parametrize(
    ("kind", "arg", "message"),
    [
        ("max_words", "sixty", "'max_words' expects a positive integer"),
        ("max_words", 0, "'max_words' expects a positive integer"),
        ("max_chars", True, "'max_chars' expects a positive integer"),
        ("contains", [], "non-empty list"),
        ("contains", True, "'contains' expects a string"),
        ("one_of", "yes", "'one_of' expects a non-empty list"),
        ("regex", "([a-z", "is invalid"),
        ("json_valid", "yes", "'json_valid' takes no argument"),
        ("json_field", "priority", "expects a mapping"),
        ("json_field", {"equals": "x"}, "needs a 'path'"),
        ("json_field", {"path": "a", "equal": "x"}, "did you mean 'equals'?"),
        ("json_field", {"path": "a", "type": "str"}, "is not a JSON type"),
        ("json_field", {"path": "a", "exists": "yes"}, "must be true or false"),
        ("json_field", {"path": "a", "exists": False, "equals": 1}, "cannot also check"),
    ],
)
def test_config_errors(kind, arg, message):
    with pytest.raises(AssertionConfigError, match=re.escape(message)):
        build_check(kind, arg)


def test_split_path_accepts_brackets_and_dots():
    assert split_path("items[0].sku") == ["items", "0", "sku"]
    assert split_path("a.b.2") == ["a", "b", "2"]


def test_labels_are_readable():
    assert build_check("json_field", {"path": "priority", "equals": "urgent"}).label() == (
        'json_field priority equals "urgent"'
    )
    assert build_check("json_field", {"path": "summary"}).label() == "json_field summary exists"
    assert build_check("contains", ["a", "b"]).label() == 'contains "a", "b"'
