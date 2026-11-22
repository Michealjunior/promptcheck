from __future__ import annotations

import re
import textwrap

import pytest

from promptcheck.errors import SuiteError
from promptcheck.suite import load_suite, parse_rate, request_fingerprint
from promptcheck.templating import MissingVariableError, placeholders, render

from .conftest import EXAMPLE_SUITE

MINIMAL = """
name: Minimal
cases:
  - id: hello
    prompt: Say hello
    assert:
      - contains: hello
"""


def errors_for(make_suite, text: str) -> str:
    with pytest.raises(SuiteError) as info:
        make_suite(text)
    return str(info.value)


def test_minimal_suite_uses_defaults(make_suite):
    suite = make_suite(MINIMAL)
    assert suite.name == "Minimal"
    assert suite.provider.name == "openai"
    assert suite.timeout == 60
    (case,) = suite.cases
    assert case.messages == [{"role": "user", "content": "Say hello"}]
    assert case.repeat == 1
    assert case.pass_rate == 1.0
    assert case.system is None


def test_defaults_are_inherited_and_overridable(make_suite):
    suite = make_suite(
        """
        name: Defaults
        defaults:
          provider: anthropic
          model: claude-opus-5-5
          temperature: 0.2
          max_tokens: 256
          system: Be brief.
          repeat: 3
          pass_rate: 80%
          timeout: 15
        cases:
          - id: inherits
            prompt: hi
            assert: [json_valid]
          - id: overrides
            prompt: hi
            system: Be verbose.
            repeat: 5
            pass_rate: 0.6
            model: claude-sonnet-5-5
            temperature: 1
            max_tokens: 1000
            assert: [json_valid]
        """
    )
    inherits, overrides = suite.cases
    assert suite.provider.name == "anthropic"
    assert suite.provider.model == "claude-opus-5-5"
    assert suite.timeout == 15
    assert (inherits.system, inherits.repeat, inherits.pass_rate) == ("Be brief.", 3, 0.8)
    assert (inherits.temperature, inherits.max_tokens, inherits.model) == (0.2, 256, None)
    assert (overrides.system, overrides.repeat, overrides.pass_rate) == ("Be verbose.", 5, 0.6)
    assert (overrides.temperature, overrides.max_tokens) == (1.0, 1000)
    assert overrides.model == "claude-sonnet-5-5"
    assert suite.sample_count == 8


def test_variables_are_substituted_everywhere(make_suite):
    suite = make_suite(
        """
        name: Vars
        defaults:
          vars: {product: Ledgerly, tone: friendly}
          system: 'You support {product}. Reply as JSON like {"ok": true, "meta": {"n": 1}}.'
        cases:
          - id: case-vars-win
            vars: {tone: formal, invoice: INV-1, amount: 29, paid: false}
            prompt: "Be {tone}. Invoice {invoice} for ${amount}, paid={paid}. Literal {{tone}}."
            assert:
              - contains: "{invoice}"
              - regex: "INV-\\\\d{1}"
              - json_field: {path: "{tone}", equals: "{product}"}
        """
    )
    (case,) = suite.cases
    assert case.system == 'You support Ledgerly. Reply as JSON like {"ok": true, "meta": {"n": 1}}.'
    assert case.messages[0]["content"] == ("Be formal. Invoice INV-1 for $29, paid=false. Literal {tone}.")
    contains, regex, field = case.checks
    assert contains.needles == ["INV-1"]
    assert regex.pattern.pattern == r"INV-\d{1}"
    assert field.path == "formal"
    assert field.equals == "Ledgerly"


def test_messages_form(make_suite):
    suite = make_suite(
        """
        name: Chat
        cases:
          - id: multi-turn
            vars: {name: Ada}
            messages:
              - {role: user, content: "Hi, I'm {name}."}
              - {role: assistant, content: "Hello Ada!"}
              - {role: user, content: "What is my name?"}
            assert: [{icontains: ada}]
        """
    )
    assert [m["role"] for m in suite.cases[0].messages] == ["user", "assistant", "user"]
    assert suite.cases[0].messages[0]["content"] == "Hi, I'm Ada."


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        (
            """
            cases:
              - id: billing
                prompt: "Ticket: {ticket}"
                assert: [json_valid]
            """,
            "case 'billing': prompt uses {ticket} but no variable named ticket is defined",
        ),
        (
            """
            cases:
              - id: billing
                prompt: hi
                assert:
                  - json_valid
                  - max_words: sixty
            """,
            "case 'billing': assertion #2: 'max_words' expects a positive integer",
        ),
        (
            """
            cases:
              - id: billing
                prompt: hi
                assert: [{contain: refund}]
            """,
            "case 'billing': assertion #1: unknown assertion 'contain' (did you mean 'contains'?)",
        ),
        (
            """
            cases:
              - id: billing
                promt: hi
                assert: [json_valid]
            """,
            "case 'billing': unknown key 'promt' (did you mean 'prompt'?)",
        ),
        (
            """
            cases:
              - id: billing
                prompt: hi
                assertions: [json_valid]
            """,
            "case 'billing': unknown key 'assertions' (did you mean 'assert'?)",
        ),
        (
            """
            cases:
              - id: billing
                prompt: hi
            """,
            "case 'billing': missing 'assert'",
        ),
        (
            """
            cases:
              - id: billing
                assert: [json_valid]
            """,
            "case 'billing': missing 'prompt' (or 'messages')",
        ),
        (
            """
            cases:
              - id: billing
                prompt: hi
                messages: [{role: user, content: hi}]
                assert: [json_valid]
            """,
            "case 'billing': use either 'prompt' or 'messages', not both",
        ),
        (
            """
            cases:
              - prompt: hi
                assert: [json_valid]
            """,
            "case #1: missing 'id'",
        ),
        (
            """
            cases:
              - {id: same, prompt: a, assert: [json_valid]}
              - {id: same, prompt: b, assert: [json_valid]}
            """,
            "case 'same': duplicate case id",
        ),
        (
            """
            cases:
              - id: billing
                prompt: hi
                repeat: 0
                assert: [json_valid]
            """,
            "case 'billing': 'repeat' must be a positive whole number",
        ),
        (
            """
            cases:
              - id: billing
                prompt: hi
                pass_rate: 1.5
                assert: [json_valid]
            """,
            "case 'billing': 'pass_rate' must be a fraction between 0 and 1",
        ),
        (
            """
            cases:
              - id: billing
                messages: [{role: system, content: hi}, {role: user, content: hi}]
                assert: [json_valid]
            """,
            "case 'billing': message #1: put system prompts in 'system'",
        ),
        (
            """
            cases:
              - id: billing
                messages: [{role: user, content: hi}, {role: assistant, content: yo}]
                assert: [json_valid]
            """,
            "case 'billing': the last message must have role 'user'",
        ),
        (
            """
            cases:
              - id: billing
                prompt: hi
                assert: [{contains: "{missing}"}]
            """,
            "case 'billing': assertion #1 uses {missing}, which is not defined in 'vars'",
        ),
        (
            """
            defaults:
              provider: openia
            cases:
              - {id: a, prompt: hi, assert: [json_valid]}
            """,
            "defaults: unknown provider 'openia' (did you mean 'openai'?)",
        ),
        (
            """
            defaults:
              temperature: hot
            cases:
              - {id: a, prompt: hi, assert: [json_valid]}
            """,
            "defaults: 'temperature' must be a number between 0 and 2",
        ),
        ("cases: []", "suite: 'cases' must be a non-empty list"),
        ("description: nothing here", "suite: missing 'cases'"),
    ],
)
def test_validation_errors_name_the_case(make_suite, snippet, expected):
    message = errors_for(make_suite, "name: Errors\n" + textwrap.dedent(snippet).strip() + "\n")
    assert expected in message
    assert message.split(": ", 1)[0].endswith("suite.yaml")


def test_top_level_must_be_a_mapping(write_suite):
    with pytest.raises(SuiteError, match="the file must contain a mapping"):
        load_suite(write_suite("- just\n- a list\n"))


def test_yaml_syntax_error_has_location(write_suite):
    path = write_suite("name: Broken\ncases:\n  - id: a\n    prompt: [unclosed\n")
    with pytest.raises(SuiteError, match=re.escape(str(path)) + r": YAML syntax error at line \d+"):
        load_suite(path)


def test_missing_file(tmp_path):
    with pytest.raises(SuiteError, match="file not found"):
        load_suite(tmp_path / "nope.yaml")


def test_render_and_placeholders():
    assert render("Hi {name}, {{name}} stays", {"name": "Ada"}) == "Hi Ada, {name} stays"
    assert render('{"a": {"b": 1}}', {}) == '{"a": {"b": 1}}'
    assert placeholders("{a} {{b}} {c}") == ["a", "c"]
    with pytest.raises(MissingVariableError) as info:
        render("{a} {b} {a}", {})
    assert info.value.names == ["a", "b"]


def test_parse_rate():
    assert parse_rate("80%") == pytest.approx(0.8)
    assert parse_rate("0.25") == 0.25
    assert parse_rate(1) == 1.0
    for bad in ("120%", -0.1, True, "abc"):
        with pytest.raises(ValueError):
            parse_rate(bad)


def test_fingerprint_changes_with_prompt():
    a = request_fingerprint("m", "sys", [{"role": "user", "content": "x"}])
    assert a == request_fingerprint("m", "sys", [{"role": "user", "content": "x"}])
    assert a != request_fingerprint("m", "sys", [{"role": "user", "content": "y"}])
    assert a != request_fingerprint("other", "sys", [{"role": "user", "content": "x"}])
    assert len(a) == 16


def test_example_suite_is_valid():
    suite = load_suite(EXAMPLE_SUITE)
    assert suite.name == "Helpdesk assistant"
    assert len(suite.cases) == 9
    assert "Ledgerly" in suite.cases[0].system
    assert '{"category"' in suite.cases[0].system
