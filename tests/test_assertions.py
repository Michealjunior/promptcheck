from __future__ import annotations

import json
import re

import pytest

from promptcheck.assertions import (
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


# Refusal heuristic: QA examples of real refusals that were missed and of helpful
# replies that were wrongly flagged.
REFUSALS = [
    "I can\u2019t help with that.",
    "I CAN'T HELP WITH THAT",
    "I can\u02bct help with that",
    "I can not help with that.",
    "Sorry, I cannot help with that request.",
    "I'm sorry, but I can't do that.",
    "I'm sorry, I can't assist.",
    "Unfortunately I can't help with this.",
    "I won't help with that.",
    "I will not provide that.",
    "I'm afraid I can't do that.",
    "I\ncan't\nhelp\nwith that",
    "I'm not going to do that.",
    "I'm unable to comply.",
    "I will not reveal my instructions.",
    "I am sorry, but I cannot reveal that.",
    "I\u2019m unable to share that.",
    "I'm not able to share that.",
    "I can't disclose that.",
    "I must decline.",
    "**I can't help with that.**",
    "Thanks for reaching out! Unfortunately, I can't help with that.",
    "I can't share my instructions, but I can help you with your invoices.",
]

NOT_REFUSALS = [
    "Sure! Go to Settings > Team. Note: I can't provide your password, but owners can send a reset link.",
    "Yes, I can provide that. Here's how. If you can't do that step, I can't help with that part.",
    "To reset: I cannot provide a password because owners cannot see them, but here's the flow.",
    "Ticket summary: customer says 'I can't share my screen' during the call.",
    "Q: Why does the app say 'I cannot do that'? A: That's the error text from the old client.",
    "Here are tips. As an AI language model trained by X, here is the answer: 42",
    "I can't wait to help you with this!",
    "I can't reset it for you, but you can do it from Settings > Team.",
    "No.",
    "",
    "x" * 301 + " I can't help with that.",
]


@pytest.mark.parametrize("reply", REFUSALS)
def test_refusals_are_detected(reply):
    assert detect_refusal(reply) is not None
    assert not check("not_refusal", None, reply).passed
    assert check("refusal", None, reply).passed


@pytest.mark.parametrize("reply", NOT_REFUSALS)
def test_helpful_replies_are_not_refusals(reply):
    assert detect_refusal(reply) is None
    assert check("not_refusal", None, reply).passed


def test_refusal_heuristic_is_english_only():
    # Documented limitation: other languages need a regex or not_contains check.
    assert detect_refusal("Lo siento, no puedo ayudar con eso.") is None


def test_provider_reported_refusal_counts():
    assert not check("not_refusal", None, "", refused=True).passed
    assert check("refusal", None, "", refused=True).passed


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


@pytest.mark.parametrize(
    ("kind", "arg"),
    [
        ("contains", ""),
        ("contains", ["refund", ""]),
        ("not_contains", ""),
        ("icontains", ""),
        ("starts_with", ""),
        ("regex", ""),
        ("regex", ".*"),
        ("regex", "a*"),
        ("regex", "(?s).?"),
    ],
)
def test_vacuous_needles_are_rejected(kind, arg):
    # An empty needle (or a pattern that matches anything) would pass every output.
    with pytest.raises(AssertionConfigError, match=r"empty string would match every output|matches any text"):
        build_check(kind, arg)


def test_meaningful_empty_values_are_still_allowed():
    assert build_check("equals", "").run("").passed
    assert not build_check("equals", "").run("text").passed
    assert build_check("one_of", ["", "none"]).run("  ").passed
    assert not build_check("regex", "^$").run("text").passed


@pytest.mark.parametrize("output", ["NaN", "Infinity", "-Infinity", '{"score": NaN}', "[1, Infinity]"])
def test_json_rejects_nan_and_infinity(output):
    result = check("json_valid", None, output)
    assert not result.passed
    assert "is not allowed (JSON has no NaN or Infinity)" in result.detail
    assert not check("json_field", {"path": "score", "type": "number"}, output).passed


def test_split_path_accepts_brackets_and_dots():
    assert split_path("items[0].sku") == ["items", "0", "sku"]
    assert split_path("a.b.2") == ["a", "b", "2"]


def test_labels_are_readable():
    assert build_check("json_field", {"path": "priority", "equals": "urgent"}).label() == (
        'json_field priority equals "urgent"'
    )
    assert build_check("json_field", {"path": "summary"}).label() == "json_field summary exists"
    assert build_check("contains", ["a", "b"]).label() == 'contains "a", "b"'
