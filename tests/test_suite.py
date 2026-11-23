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
            "case 'billing': prompt: {ticket} is not defined in 'vars'",
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
            "case 'billing': 'pass_rate' must be greater than 0 and at most 1",
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
            "case 'billing': assertion #1: {missing} is not defined in 'vars'",
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
    for bad in ("120%", -0.1, True, "abc", 0, "0%", "0", float("nan"), "nan"):
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


# ----------------------------------------------------------------- regression tests (QA)


def test_duplicate_assert_key_is_rejected_with_case_and_lines(write_suite):
    # M1: yaml.safe_load kept only the second 'assert', so the first list never ran.
    path = write_suite(
        """
        name: Duplicate keys
        cases:
          - id: refund
            prompt: Can I get a refund?
            assert:
              - icontains: refund
              - not_refusal
            assert:
              - contains: "30 days"
        """
    )
    with pytest.raises(SuiteError) as info:
        load_suite(path)
    message = str(info.value)
    assert "case 'refund': duplicate key 'assert' at line 8 (first defined at line 5)" in message
    assert "keep only the last one" in message


@pytest.mark.parametrize(
    ("text", "scope"),
    [
        ("name: a\nname: b\ncases: []\n", "suite: duplicate key 'name' at line 2"),
        ("defaults:\n  model: a\n  model: b\ncases: []\n", "defaults: duplicate key 'model' at line 3"),
        (
            "cases:\n  - id: x\n    prompt: hi\n    vars: {a: 1, a: 2}\n    assert: [json_valid]\n",
            "case 'x': duplicate key 'a' at line 4",
        ),
        (
            "cases:\n  - prompt: hi\n    prompt: again\n    assert: [json_valid]\n",
            "case #1: duplicate key 'prompt' at line 3",
        ),
    ],
)
def test_duplicate_keys_anywhere_are_rejected(write_suite, text, scope):
    with pytest.raises(SuiteError, match=re.escape(scope)):
        load_suite(write_suite(text))


def test_merge_keys_may_be_overridden(make_suite):
    suite = make_suite(
        """
        name: Anchors
        cases:
          - &base {id: a, prompt: hi, assert: [json_valid]}
          - <<: *base
            id: b
            prompt: hello
        """
    )
    assert [c.id for c in suite.cases] == ["a", "b"]
    assert suite.cases[1].messages[0]["content"] == "hello"


def test_null_variable_is_an_error_naming_var_and_case(make_suite):
    # M2: a var declared without a value used to render as "" and pass vacuously.
    message = errors_for(
        make_suite,
        """
        name: Null var
        defaults:
          vars:
            invoice_id:
        cases:
          - id: summary
            prompt: "Summarise invoice {invoice_id}"
            assert: [{min_words: 1}]
        """,
    )
    assert "case 'summary': prompt: {invoice_id} has no value (null)" in message


def test_null_variable_in_assertion_is_an_error(make_suite):
    message = errors_for(
        make_suite,
        """
        name: Null var
        cases:
          - id: summary
            prompt: hi
            vars: {invoice_id: null}
            assert: [{contains: "{invoice_id}"}]
        """,
    )
    assert "case 'summary': assertion #1: {invoice_id} has no value (null)" in message


def test_empty_variable_that_empties_a_needle_is_an_error(make_suite):
    message = errors_for(
        make_suite,
        """
        name: Empty var
        cases:
          - id: ticket
            prompt: "Ticket {ticket}"
            vars: {ticket: ""}
            assert: [{icontains: "{ticket}"}]
        """,
    )
    assert "case 'ticket': assertion #1: 'icontains' needs a non-empty string" in message
    assert "(after substituting variables)" in message


def test_empty_needle_is_rejected_at_validation(make_suite):
    message = errors_for(
        make_suite,
        """
        name: Empty needles
        cases:
          - id: needles
            prompt: hi
            assert:
              - contains: ""
        """,
    )
    assert "case 'needles': assertion #1: 'contains' needs a non-empty string" in message


def test_unicode_variable_names_are_substituted(make_suite):
    suite = make_suite(
        """
        name: Unicode vars
        defaults:
          vars: {café: Ledgerly}
        cases:
          - id: a
            prompt: "Hi {名前}, welcome to {café}. Literal {{名前}}."
            vars: {名前: Taro}
            assert: [{contains: "{café}"}]
        """
    )
    assert suite.cases[0].messages[0]["content"] == "Hi Taro, welcome to Ledgerly. Literal {名前}."
    assert suite.cases[0].checks[0].needles == ["Ledgerly"]


def test_invalid_variable_names_are_rejected(make_suite):
    message = errors_for(
        make_suite, "name: x\ncases:\n  - {id: a, prompt: hi, vars: {1abc: x}, assert: [json_valid]}\n"
    )
    assert "variable name '1abc' must be an identifier" in message


@pytest.mark.parametrize("rate", ["0", "0%", "0.0"])
def test_zero_pass_rate_is_rejected(make_suite, rate):
    message = errors_for(
        make_suite, f"name: x\ncases:\n  - {{id: a, prompt: hi, pass_rate: {rate}, assert: [json_valid]}}\n"
    )
    assert "'pass_rate' must be greater than 0 and at most 1" in message


def test_utf16_suite_has_a_clear_error(tmp_path):
    path = tmp_path / "utf16.yaml"
    path.write_bytes("name: x\ncases: []\n".encode("utf-16"))
    with pytest.raises(SuiteError, match="looks like UTF-16 or UTF-32 text; save the suite file as UTF-8"):
        load_suite(path)


def test_latin1_suite_has_a_clear_error(tmp_path):
    path = tmp_path / "latin1.yaml"
    path.write_bytes("name: café\ncases: []\n".encode("latin-1"))
    with pytest.raises(SuiteError, match=r"is not valid UTF-8 \(byte 0xe9 on line 1\)"):
        load_suite(path)


def test_utf8_bom_is_accepted(tmp_path):
    path = tmp_path / "bom.yaml"
    path.write_bytes(b"\xef\xbb\xbf" + MINIMAL.encode())
    assert load_suite(path).name == "Minimal"


def test_directory_instead_of_suite(tmp_path):
    with pytest.raises(SuiteError, match="is a directory, expected a suite file"):
        load_suite(tmp_path)


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ('""', "must be an http(s) URL with a host, got an empty value"),
        ("ftp://example.com", "must be an http(s) URL with a host"),
        ("localhost:8000/v1", "must be an http(s) URL with a host"),
        ("https://user:pw@example.com", "must not contain credentials"),
        ("https://example.com/v1?x=1", "must not contain a query string"),
    ],
)
def test_suite_base_url_is_validated(make_suite, url, message):
    text = f"name: x\ndefaults:\n  base_url: {url}\ncases:\n  - {{id: a, prompt: hi, assert: [json_valid]}}\n"
    with pytest.raises(SuiteError, match=re.escape(f"defaults: 'base_url' {message}")):
        make_suite(text)
