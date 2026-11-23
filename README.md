# promptcheck

**Regression tests for LLM prompts. Catch silent breakage before your users do.**

[![Tests: 367 passing](https://img.shields.io/badge/tests-367%20passing-00ff9d.svg)](#testing)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-00ff9d.svg)](LICENSE)
[![Python 3.10 | 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-00e1ff.svg)](pyproject.toml)
[![Live site](https://img.shields.io/badge/live%20site-michealjunior.github.io%2Fpromptcheck-0a1218.svg)](https://michealjunior.github.io/promptcheck/)

![The promptcheck HTML report for the bundled example suite: 6 of 9 cases passing, with the failing outage-triage case expanded to show the failed assertion and the model output](docs/screenshot.png)

<sub>The HTML report for the bundled example suite, generated offline from recorded outputs. [Open the live sample report](https://michealjunior.github.io/promptcheck/sample-report.html).</sub>

## The problem

Teams change a prompt, swap a model or pick up a new provider version, and the outputs quietly drift:

- **JSON stops parsing** because the model starts wrapping it in a Markdown fence.
- **Required fields vanish** or change value: an outage is suddenly `"high"` instead of `"urgent"`.
- **Answers grow longer** and no longer fit the UI they were written for.
- **The model starts refusing** perfectly legitimate requests after a safety update.

Nothing throws an exception, so nothing fails. Without tests in CI, nobody notices until users complain.

## What it does

promptcheck is a small command-line tool that treats prompts like code:

- **YAML test suites.** Defaults (provider, model, temperature, system prompt, repeat) plus cases with a `prompt` or a multi-turn `messages` list, `{var}` substitution and assertions.
- **14 assertion types** for text, structure, length and behaviour, including `json_field` with dotted paths and a documented refusal heuristic. Every failure explains itself (`got "high"`, `104 words, limit is 60`).
- **Flakiness measurement.** Sample a case `repeat` times and require a `pass_rate`, such as 4 out of 5.
- **Providers:** any OpenAI-compatible Chat Completions API (OpenAI, OpenRouter, Groq, Ollama, vLLM and others), the Anthropic Messages API, and **replay** of recorded outputs so CI is deterministic and free. `--record` saves live outputs for replay.
- **A resilient runner.** Requests run on a thread pool. `--timeout` is a deadline for each whole request, and 408, 429, 5xx, dropped connections and truncated responses are retried with exponential backoff. A failing request becomes an error on that case and never aborts the run; Ctrl-C cancels queued requests and exits at once.
- **Fails loudly, never vacuously.** Duplicate YAML keys, undefined or null variables, empty assertion values, `NaN` in "JSON" and tool-call replies with no text are errors, not silent passes. A case with no passing sample never passes.
- **Reports:** a coloured terminal summary (plain when piped or when `NO_COLOR` is set), JUnit XML, Markdown for pull-request comments and a single-file HTML report. Model output is escaped everywhere: control sequences in the terminal, HTML and Markdown in PR comments.
- **CI-friendly exit codes.** Exit code 1 when cases fail, tunable with `--min-pass-rate`.
- **One runtime dependency** (PyYAML). HTTP goes through the standard library's `urllib`. API keys are read from environment variables only, are scrubbed from error messages and outputs, and are never sent anywhere but the base URL you configured: redirects are not followed.

## Quick start

### Install

promptcheck needs Python 3.10 or newer. The easiest way to install it as a command is [uv](https://docs.astral.sh/uv/), which also fetches a suitable Python if yours is too old (macOS ships 3.9 and no `pip` command):

```bash
uv tool install git+https://github.com/Michealjunior/promptcheck
promptcheck --version
```

Alternatives: `pipx install git+https://github.com/Michealjunior/promptcheck`, or inside a Python 3.10+ virtual environment, `python3 -m pip install git+https://github.com/Michealjunior/promptcheck`. To try it once without installing: `uvx --from git+https://github.com/Michealjunior/promptcheck promptcheck --help`.

### Try the example offline

The repository ships a realistic suite for a helpdesk assistant and a recording of its outputs, so you can see a full run without an API key:

```bash
git clone https://github.com/Michealjunior/promptcheck
cd promptcheck
promptcheck run examples/support-triage.yaml \
  --provider replay --replay examples/recordings.json \
  --html report.html --junit junit.xml --markdown summary.md
```

```text
promptcheck 0.1.0 · Helpdesk assistant · replay (examples/recordings.json) · gpt-4.1-mini
9 cases, 13 samples

PASS triage-double-charge         1/1
FAIL triage-checkout-outage       0/1
     ✗ json_field priority equals "urgent": got "high"
     │ {"category": "technical", "priority": "high", "summary": "Invoice payment links have …
FAIL triage-feature-request       0/1
     ✗ json_valid: output is wrapped in a Markdown code fence, so it is not valid JSON
     ✗ json_field category equals "feature_request": cannot check: output is not valid JSON
     ✗ json_field priority equals "low": cannot check: output is not valid JSON
PASS triage-locked-account        1/1
PASS triage-ambiguous-ticket      4/5  80% (needs 80%)
     ✗ sample 3: json_field category one_of "billing", "technical": got "bug"
PASS summarise-refund-thread      1/1
FAIL summarise-escalation-thread  0/1
     ✗ max_words 60: 104 words, limit is 60
PASS reply-admin-password-reset   1/1
PASS reply-prompt-injection       1/1
────────────────────────────────────────────────────────────
FAILED  6 passed, 3 failed, 1 flaky  ·  6/9 cases (66.7%)  ·  1 ms
```

(Output excerpts trimmed for width.) The run exits with code 1 because three cases fail on purpose. The recordings in [`examples/recordings.json`](examples/recordings.json) are **hand-written fixtures** that mimic typical model output, including three realistic regressions; they were not captured from a live model. Run the suite live with `--record` to replace them with real outputs.

### Write your own suite

```yaml
# prompts/suite.yaml
name: Support bot
defaults:
  provider: openai
  model: gpt-4.1-mini
  temperature: 0
  system: |
    You are the support assistant for {product}. Answer in at most two sentences.
  vars:
    product: Ledgerly

cases:
  - id: refund-window
    prompt: Can I still get a refund {days} days after paying?
    vars: {days: 45}
    assert:
      - icontains: refund
      - max_words: 60
      - not_refusal

  - id: classify-sentiment
    prompt: "Reply with one word, positive, neutral or negative: {review}"
    vars: {review: "The export button finally works. Thank you!"}
    repeat: 5
    pass_rate: 0.8
    assert:
      - one_of: [positive, neutral, negative]
      - equals: positive
```

### Run it

```bash
promptcheck validate prompts/suite.yaml           # check the file without calling a model

export OPENAI_API_KEY=...                         # read from the environment, never printed
promptcheck run prompts/suite.yaml --record prompts/recordings.json

promptcheck run prompts/suite.yaml --replay prompts/recordings.json   # offline, deterministic
```

Commit `recordings.json` next to the suite. When you change a prompt, re-record and review the diff like any other change; replay warns when a recording no longer matches the prompt, system prompt or model it was recorded with.

### Command-line options

`promptcheck run SUITE [options]`

| Option | Purpose |
|---|---|
| `--provider NAME` | Override the suite's provider: `openai`, `openrouter`, `groq`, `ollama`, `anthropic` or `replay`. |
| `--model NAME` | Override the default model. |
| `--base-url URL` | Override the API base URL (OpenAI-compatible or Anthropic). Must be an `http(s)` URL with a host; an empty or invalid value is an error and is never replaced by the default endpoint. |
| `--replay PATH` | Serve recorded outputs from `PATH` (implies `--provider replay`). |
| `--record PATH` | Save live outputs to `PATH` for later replay. Existing cases in the file are kept; a file that is not a valid recording (for example one with merge-conflict markers) is never overwritten. |
| `--junit PATH` / `--markdown PATH` / `--html PATH` | Write reports. |
| `--case PATTERN` | Only run cases whose id matches a glob such as `triage-*`. Repeatable. |
| `--repeat N` | Sample every case `N` times (1 to 100). |
| `-j`, `--concurrency N` | Parallel requests, 1 to 64 (default 4). |
| `--timeout SECONDS` | Deadline for each request attempt, including reading the whole response, up to 3600 (default: the suite's `timeout`, or 60). Retries start a new deadline. |
| `--retries N` | Retries on 408, 429, 5xx, timeouts and network errors, including connections closed mid-response, 0 to 10 (default 3). |
| `--min-pass-rate RATE` | Fraction of cases that must pass for exit code 0, above 0 and at most 1, as `0.9` or `90%` (default 1.0). |
| `--no-color`, `-v`/`--verbose`, `-q`/`--quiet` | Output control. `FORCE_COLOR=1` forces colour in CI logs. |

`promptcheck validate SUITE` loads and validates a suite and prints a summary.

Report and recording paths are checked before any request is sent. Results are printed before any file is written, so a write failure never hides them, and files are written atomically (a temporary file, then a rename). If a recording cannot be written, its outputs are saved to a temporary file instead and the path is printed.

Exit codes: `0` all good; `1` cases failed (or the pass rate is below `--min-pass-rate`); `2` usage or configuration error (invalid suite, missing API key, unwritable output), a report that could not be written, or an internal error in promptcheck itself (reported as a bug, not as a failing case); `130` interrupted with Ctrl-C (queued requests are cancelled and no files are written).

## Writing tests

A suite has a `name`, optional `description`, optional `defaults` and a list of `cases`.

### Defaults

| Key | Meaning |
|---|---|
| `provider` | `openai` (default), `openrouter`, `groq`, `ollama`, `anthropic` or `replay`. |
| `model` | Default model for every case. |
| `base_url`, `api_key_env` | Point the provider at another `http(s)` endpoint or read the key from another variable. |
| `anthropic_version` | Anthropic API version header (default `2026-10-21`). |
| `temperature`, `max_tokens` | Sampling settings. They are only sent when set. |
| `system` | Default system prompt. |
| `repeat`, `pass_rate` | Default samples per case (1 to 100) and the fraction that must pass, above 0 and at most 1 (`0.8` or `"80%"`). |
| `timeout` | Deadline in seconds for each request attempt, including the response body (default 60). |
| `vars` | Variables available to every case. |

### Cases

| Key | Meaning |
|---|---|
| `id` | Required, unique. Used in reports and recordings. |
| `prompt` | A single user message. |
| `messages` | Instead of `prompt`: a list of `{role: user \| assistant, content: ...}` turns ending with `user`. |
| `system` | Overrides the default system prompt for this case. |
| `vars` | Case variables, merged over `defaults.vars`. |
| `assert` | Required list of assertions; every one must pass for a sample to pass. |
| `repeat`, `pass_rate` | Sample the case several times and require a pass rate. The case passes when at least one sample passes and `passed samples / repeat >= pass_rate`. Errors count as failed samples. |
| `model`, `temperature`, `max_tokens`, `description` | Per-case overrides and documentation. |

**Variables.** `{name}` is replaced in `prompt`, `messages`, `system` and the values of text assertions (`regex` is never templated). Only `{identifier}` is a placeholder, so JSON examples such as `{"category": "billing"}` are left alone; write `{{name}}` for a literal `{name}`. Names follow Python identifier rules and may use non-ASCII letters (`{名前}`). Using a variable that is undefined, or defined without a value (`invoice_id:` with nothing after it), is a validation error that names the variable and the case. An empty-string variable is allowed in prompts, but not where it would leave an assertion with nothing to check.

**File format.** Suites must be UTF-8 (a byte-order mark is fine); other encodings get a clear error. Duplicate keys are rejected with both line numbers, because YAML would otherwise silently keep only the last one, for example dropping the first of two `assert:` blocks.

**Validation.** Mistakes are reported before any request is made, with the file, case id and assertion number, and a suggestion when the name is close:

```text
suite.yaml: case 'billing': assertion #2: unknown assertion 'contain' (did you mean 'contains'?)
```

### Assertion reference

Write each assertion as a single-key mapping. Assertions without an argument can also be written as a bare word (`- json_valid`). Text comparisons ignore surrounding whitespace where noted. Values that could never fail are rejected: an empty string for `contains`, `not_contains`, `icontains`, `starts_with` or `regex`, and a `regex` that matches any text, such as `.*`.

| Assertion | Example | Passes when |
|---|---|---|
| `contains` | `contains: refund` or `contains: [refund, "30 days"]` | The output contains every string (case-sensitive). |
| `not_contains` | `not_contains: ["As an AI", "system prompt"]` | The output contains none of the strings. Failures show the surrounding text. |
| `icontains` | `icontains: [settings, team]` | Like `contains`, ignoring case. |
| `regex` | `regex: "INV-\\d{4}"` | Python `re.search` finds a match. Use inline flags such as `(?i)` or `(?m)`. |
| `equals` | `equals: positive` | The trimmed output equals the value exactly. |
| `starts_with` | `starts_with: "Dear "` | The output, ignoring leading whitespace, starts with the value. |
| `one_of` | `one_of: [positive, neutral, negative]` | The trimmed output equals one of the values. |
| `json_valid` | `json_valid` | The trimmed output parses as strict JSON (`NaN` and `Infinity` are rejected). A Markdown code fence is reported as such. |
| `json_field` | `json_field: {path: priority, equals: urgent}` | The output is JSON and the value at `path` satisfies every given check (see below). |
| `max_words` | `max_words: 60` | At most N whitespace-separated words. |
| `min_words` | `min_words: 15` | At least N words. |
| `max_chars` | `max_chars: 400` | At most N characters after trimming. |
| `not_refusal` | `not_refusal` | The reply does not look like a refusal (heuristic below) and the provider did not report one. |
| `refusal` | `refusal` | The reply looks like a refusal, or the provider reported one. Useful for safety and prompt-injection cases. |

**`json_field` options.** `path` is required and uses dots, with numbers or `[n]` for array indexes (`customer.tier`, `items.0.sku`, `items[0].sku`). Combine any of:

| Option | Meaning |
|---|---|
| `equals` | Strict JSON equality. `true` never equals `1`; `1` equals `1.0`. Objects and arrays compare deeply. |
| `one_of` | The value equals one of the listed values. |
| `type` | `string`, `number`, `integer`, `boolean`, `array`, `object` or `null`. |
| `exists` | `true` (the default when no other check is given) or `false` to require that the path is absent. |

**Refusal heuristic.** `not_refusal` and `refusal` look at how the reply *opens*. A reply counts as a refusal when its first sentence (or its second, after a short opener such as "Thanks for asking!") starts with a first-person refusal: optionally an apology ("I'm sorry, but", "Unfortunately,", "I'm afraid") followed by "I can't / cannot / won't / will not / am unable to / am not able to / am not going to" and a verb such as help, assist, provide, share, do, comply, reveal or disclose, or by "I must decline". Apostrophe variants and line breaks are normalised. Refusal wording later in a helpful answer, inside a quotation, or followed by a redirection ("I can't reset it for you, but you can …") does not count. The pattern is `detect_refusal` in [`src/promptcheck/assertions.py`](src/promptcheck/assertions.py). Limits: it is English-only, it only reads the first 300 characters, and unusual phrasings can slip through; add `regex` or `not_contains` checks for your own wording. A provider-reported refusal (Anthropic `stop_reason: "refusal"`, an OpenAI `refusal` message or a content-filter stop) always counts.

## Providers

| Provider | Endpoint | API key variable | Notes |
|---|---|---|---|
| `openai` | `https://api.openai.com/v1/chat/completions` | `OPENAI_API_KEY` | Sends `max_completion_tokens` to api.openai.com and `max_tokens` to other servers. |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | OpenAI-compatible preset. |
| `groq` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | OpenAI-compatible preset. |
| `ollama` | `http://localhost:11434/v1` | none | OpenAI-compatible preset for a local Ollama server. |
| `anthropic` | `https://api.anthropic.com/v1/messages` | `ANTHROPIC_API_KEY` | Messages API, `anthropic-version: 2026-10-21` by default. |
| `replay` | a local JSON file | none | Serves recorded outputs; needs `--replay PATH`. |

Any other OpenAI-compatible server (vLLM, LM Studio, a gateway) works with `provider: openai` and `base_url`. With a custom `base_url` the key is optional unless you name one with `api_key_env`:

```yaml
defaults:
  provider: openai
  base_url: http://localhost:8000/v1
  model: meta-llama/Llama-3.1-8B-Instruct
```

Anthropic example:

```yaml
defaults:
  provider: anthropic
  model: claude-opus-5-5
  max_tokens: 2048
```

Notes on Anthropic: `base_url` may be given with or without a trailing `/v1`. The Messages API requires `max_tokens`, so promptcheck sends 4096 when none is set. API version `2026-10-21` rejects sampling parameters, so a `temperature` in the suite is not sent and the run prints a note saying so; set `anthropic_version: "2023-06-01"` to send it to older models that still accept it. promptcheck deliberately does not enable server-side model fallbacks, because a regression test must measure the model it names.

**Replies without text.** A response that contains only tool calls, or no message content at all, becomes an error on that case instead of an empty output that might pass. Text plus tool calls is graded as text, with a warning.

**Redirects.** promptcheck never follows HTTP redirects for API calls: urllib would turn the POST into a GET without its body and forward the `Authorization` header to the new host. A redirect is reported as an error that names the target, so you can set `base_url` to the final address.

**Recording format.** `--record` writes a versioned JSON file (`promptcheck-recording/1`) with one entry per sample: the output, the model that produced it and a hash of the request. Replay serves sample *n* of a case from entry *n*, reports a per-case error when a case or sample is missing, and warns when the hash shows the prompt, system prompt or model changed. For hand-written fixtures a shorthand also works: `{"case-id": "output"}` or `{"case-id": ["sample 1", "sample 2"]}`.

## Use it in CI

Replay recorded outputs on every pull request: no secrets, no token spend, the same answer every time.

```yaml
# .github/workflows/prompt-tests.yml
name: Prompt tests
on: [pull_request]

jobs:
  promptcheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install git+https://github.com/Michealjunior/promptcheck
      - run: >-
          promptcheck run prompts/suite.yaml
          --replay prompts/recordings.json
          --junit reports/junit.xml
          --markdown reports/summary.md
          --html reports/report.html
      - if: always()
        run: cat reports/summary.md >> "$GITHUB_STEP_SUMMARY"
      - if: always()
        uses: actions/upload-artifact@v4
        with:
          name: promptcheck-reports
          path: reports/
```

A failing case exits with code 1 and fails the job. The Markdown summary appears on the job page, and the JUnit and HTML reports are attached as artifacts. A complete, runnable version of this workflow lives in [`.github/workflows/example-llm-tests.yml`](.github/workflows/example-llm-tests.yml).

To catch drift on the provider's side as well, add a scheduled job that runs the same suite against the live model:

```yaml
on:
  schedule:
    - cron: "0 6 * * 1"   # every Monday
jobs:
  live:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install git+https://github.com/Michealjunior/promptcheck
      - run: promptcheck run prompts/suite.yaml --junit reports/junit.xml
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## How it works

```text
suite.yaml ──► suite loader ──► runner (thread pool) ──► provider ──► model API or recording
              validate, render      one job per sample        urllib + retries
                                          │
                                          ▼
                                 assertions per sample ──► case pass rate ──► reporters
                                                                              terminal · JUnit · Markdown · HTML
```

1. **Load.** [`suite.py`](src/promptcheck/suite.py) parses the YAML, validates every key, merges defaults, renders `{var}` placeholders and builds the assertion objects, so mistakes surface before any request is sent.
2. **Run.** [`runner.py`](src/promptcheck/runner.py) expands each case into `repeat` samples and runs them on a `ThreadPoolExecutor`. Each sample is isolated: provider errors become an error result on that sample, and an unexpected exception is recorded as an internal error, which makes the run exit with code 2. Ctrl-C cancels every queued request.
3. **Call.** Providers in [`providers/`](src/promptcheck/providers) build the request for their API and parse the response, including refusal and truncation signals. [`http.py`](src/promptcheck/providers/http.py) posts JSON with `urllib` (without following redirects), enforces the deadline while reading the body, retries 408, 429, 5xx, timeouts, connection errors and truncated bodies with exponential backoff (honouring `Retry-After`), and scrubs API keys from error messages.
4. **Grade.** [`assertions.py`](src/promptcheck/assertions.py) checks each output. A sample passes when every assertion passes; a case passes when its sample pass rate meets `pass_rate`.
5. **Report.** [`reporters/`](src/promptcheck/reporters) render the same results as terminal text, JUnit XML, Markdown and a self-contained HTML page.

**Limitations.** Requests are not streamed, so very long generations rely on the per-request deadline (raise `--timeout` for them). The refusal heuristic is English-only and reads only the opening of a reply. Replay is keyed by case id and sample number, not by prompt content, so re-record after renaming cases. Assertions are deterministic checks; there is no LLM-as-judge grading.

## Testing

```bash
uv venv --python 3.12 && uv pip install -e '.[dev]'   # or: python -m venv .venv && pip install -e '.[dev]'
ruff check . && ruff format --check .
pytest
```

The suite has 367 tests and never calls a real API: `urllib` is replaced with scripted fakes, and the end-to-end tests (redirects, slow or truncated responses, recording, Ctrl-C) talk to stub servers on `127.0.0.1`. It covers every assertion type, suite validation messages, variable substitution, replay and recording, OpenAI and Anthropic request building and response parsing, retry and backoff behaviour, key redaction, escaping in all four reporters, CLI exit codes, and a regression test for every bug found in an independent QA pass. CI runs ruff and pytest on Python 3.10, 3.11, 3.12 and 3.13.

## Licence

MIT © 2026 Ayeni Oluwaferanmi Micheal. See [LICENSE](LICENSE).

## Author

**Ayeni Oluwaferanmi Micheal**, software engineer in AI evaluation and full-stack development, based in Ibadan, Nigeria.

- Website: [michealjunior.github.io](https://michealjunior.github.io)
- LinkedIn: [linkedin.com/in/ayeni-oluwaferanmi-668ba6301](https://www.linkedin.com/in/ayeni-oluwaferanmi-668ba6301)
- GitHub: [@Michealjunior](https://github.com/Michealjunior)
