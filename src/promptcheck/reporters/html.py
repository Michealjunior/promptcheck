"""Single-file HTML report. No external assets are required; web fonts load when online."""

from __future__ import annotations

from html import escape
from pathlib import Path

from promptcheck import __version__
from promptcheck.reporters.common import duration, latency, percent
from promptcheck.runner import ERROR, FAIL, PASS, CaseResult, RunResult, SampleResult

REPO_URL = "https://github.com/Michealjunior/promptcheck"

LOGO_SVG = (
    '<svg viewBox="0 0 32 32" width="28" height="28" aria-hidden="true" focusable="false">'
    '<defs><linearGradient id="pc-g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#00ff9d"/><stop offset="1" stop-color="#00e1ff"/>'
    "</linearGradient></defs>"
    '<rect width="32" height="32" rx="8" fill="url(#pc-g)"/>'
    '<path d="M8.5 16.5l4.8 4.8L23.5 11" fill="none" stroke="#04070a" stroke-width="3.2" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='8' fill='%2300ff9d'/%3E"
    "%3Cpath d='M8.5 16.5l4.8 4.8L23.5 11' fill='none' stroke='%2304070a' stroke-width='3.2' "
    "stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E"
)

STATUS_LABEL = {PASS: "Pass", FAIL: "Fail", ERROR: "Error"}

CSS = """
:root {
  --bg: #04070a; --card: #0a1218; --raised: #0f1a22; --line: rgba(0, 255, 157, 0.14);
  --line-soft: rgba(214, 228, 236, 0.08); --text: #d6e4ec; --muted: #8193a3; --accent: #00ff9d;
  --accent-soft: rgba(0, 255, 157, 0.12); --cyan: #00e1ff; --pass: #00ff9d; --pass-bg: rgba(0, 255, 157, 0.1);
  --fail: #ff2e88; --fail-bg: rgba(255, 46, 136, 0.13); --fail-text: #ff8fbf; --warn: #ffc857;
  --warn-bg: rgba(255, 200, 87, 0.1); --radius: 18px; --radius-sm: 14px;
  --font-head: "Chakra Petch", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-body: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background-color: var(--bg); color: var(--text); font: 15px/1.6 var(--font-body);
  background-image:
    radial-gradient(1000px 460px at 88% -12%, rgba(0, 225, 255, 0.08), transparent 60%),
    radial-gradient(900px 420px at -10% -5%, rgba(0, 255, 157, 0.07), transparent 60%),
    linear-gradient(rgba(0, 255, 157, 0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 255, 157, 0.025) 1px, transparent 1px);
  background-size: auto, auto, 40px 40px, 40px 40px;
  background-repeat: no-repeat, no-repeat, repeat, repeat;
}
a { color: var(--cyan); text-underline-offset: 3px; }
a:hover { color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 6px; }
.wrap { width: min(1120px, 100% - 32px); margin: 0 auto; }
.top { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 22px 0; }
.brand { display: inline-flex; align-items: center; gap: 10px; color: var(--text); text-decoration: none;
  font: 700 19px/1 var(--font-head); letter-spacing: 0.01em; }
.brand:hover { color: var(--text); }
.top .tag { color: var(--muted); font: 500 12px/1.4 var(--font-mono); letter-spacing: 0.08em;
  text-transform: uppercase; }
h1, h2, h3 { font-family: var(--font-head); letter-spacing: 0; margin: 0; }
.hud { position: relative; }
.hud::before, .hud::after { content: ""; position: absolute; width: 14px; height: 14px; pointer-events: none;
  border: 0 solid var(--accent); opacity: 0.75; }
.hud::before { top: 10px; left: 10px; border-top-width: 2px; border-left-width: 2px; }
.hud::after { right: 10px; bottom: 10px; border-right-width: 2px; border-bottom-width: 2px; }
.hero { display: grid; grid-template-columns: 1fr auto; gap: 28px; align-items: center;
  padding: 30px; border: 1px solid var(--line); border-radius: 20px;
  background: linear-gradient(180deg, var(--card), rgba(10, 18, 24, 0.75)); }
.eyebrow, .kicker { margin: 0 0 8px; color: var(--accent); font: 600 12px/1.4 var(--font-mono);
  letter-spacing: 0.1em; text-transform: uppercase; overflow-wrap: anywhere; }
.eyebrow .path { margin-left: 8px; color: var(--muted); letter-spacing: 0; text-transform: none;
  font-weight: 500; }
.hero h1 { font-size: clamp(30px, 4vw, 42px); line-height: 1.1; font-weight: 700; }
.hero .desc { margin: 10px 0 0; color: var(--muted); max-width: 62ch; }
.facts { display: flex; flex-wrap: wrap; gap: 10px 28px; margin: 20px 0 0; padding: 0; }
.facts div { min-width: 0; }
.facts dt { color: var(--muted); font: 500 11px/1.4 var(--font-mono); text-transform: uppercase;
  letter-spacing: 0.1em; }
.facts dd { margin: 3px 0 0; font-weight: 600; overflow-wrap: anywhere; }
.facts code { font: 500 13px var(--font-mono); color: var(--cyan); }
.verdict { display: flex; align-items: center; gap: 18px; padding: 18px 22px; border-radius: var(--radius-sm);
  background: var(--raised); border: 1px solid var(--line); }
.ring { position: relative; width: 92px; height: 92px; flex: none; }
.ring svg { width: 92px; height: 92px; transform: rotate(-90deg); }
.ring .track { stroke: rgba(214, 228, 236, 0.1); }
.verdict--fail .ring .track { stroke: rgba(255, 46, 136, 0.45); }
.ring .value { stroke: var(--pass); }
.ring span { position: absolute; inset: 0; display: grid; place-items: center;
  font: 700 19px/1 var(--font-head); }
.verdict strong { display: block; font: 700 21px/1.2 var(--font-head); }
.verdict--fail strong { color: var(--fail); }
.verdict--pass strong { color: var(--pass); }
.verdict small { display: block; margin-top: 4px; color: var(--muted); font-size: 14px; }
.stats { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 16px 0 0;
  padding: 0; list-style: none; }
.stat { padding: 16px 18px; border-radius: var(--radius-sm); background: var(--card);
  border: 1px solid var(--line-soft); }
.stat b { display: block; font: 700 28px/1.1 var(--font-head); }
.stat span { color: var(--muted); font: 500 11px/1.6 var(--font-mono); letter-spacing: 0.1em;
  text-transform: uppercase; }
.stat--pass b { color: var(--pass); }
.stat--fail b { color: var(--fail); }
.stat--err b { color: var(--fail); }
.stat--cyan b { color: var(--cyan); }
.stat--warn b { color: var(--warn); }
.notes { margin: 16px 0 0; padding: 14px 18px; border-radius: var(--radius-sm); background: var(--warn-bg);
  border: 1px solid rgba(255, 200, 87, 0.3); color: var(--text); }
.notes h2 { font: 600 12px/1.4 var(--font-mono); letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--warn); margin-bottom: 6px; }
.notes ul { margin: 0; padding-left: 18px; }
.toolbar { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 12px;
  margin: 40px 0 14px; }
.toolbar .kicker { margin-bottom: 4px; }
.toolbar h2 { font-size: 24px; }
.filters { display: inline-flex; flex-wrap: wrap; gap: 6px; padding: 4px; border-radius: 999px;
  background: var(--card); border: 1px solid var(--line); }
.filters button, .toggle { appearance: none; border: 0; cursor: pointer; border-radius: 999px;
  padding: 7px 14px; font: 600 13px/1 var(--font-body); color: var(--muted); background: transparent; }
.filters button[aria-pressed="true"] { background: var(--accent); color: var(--bg); }
.filters button:hover:not([aria-pressed="true"]) { color: var(--text); }
.filters .count { opacity: 0.8; margin-left: 5px; font-variant-numeric: tabular-nums; }
.toggle { border: 1px solid var(--line); color: var(--text); }
.toggle:hover { border-color: var(--accent); color: var(--accent); }
.tools { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.no-js .tools { display: none; }
.cases { display: grid; gap: 12px; }
.cases[data-filter="failing"] .case[data-status="pass"],
.cases[data-filter="passing"] .case:not([data-status="pass"]) { display: none; }
.case { border-radius: var(--radius); background: var(--card); border: 1px solid var(--line-soft);
  border-left: 3px solid rgba(0, 255, 157, 0.55); overflow: hidden; }
.case[data-status="fail"], .case[data-status="error"] { border-color: rgba(255, 46, 136, 0.28);
  border-left-color: var(--fail); }
.case > summary { list-style: none; cursor: pointer; display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto auto; align-items: start; gap: 6px 14px;
  padding: 16px 20px; }
.case > summary::-webkit-details-marker { display: none; }
.case > summary:hover { background: rgba(0, 255, 157, 0.03); }
.pill { display: inline-flex; align-items: center; justify-content: center; gap: 6px; min-width: 66px;
  padding: 4px 10px; border-radius: 999px; font: 600 11.5px/1.3 var(--font-mono); letter-spacing: 0.08em;
  text-transform: uppercase; }
.pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.pill--pass { color: var(--pass); background: var(--pass-bg); }
.pill--fail { color: var(--fail-text); background: var(--fail-bg); }
.pill--error { color: var(--fail-text); background: transparent; box-shadow: inset 0 0 0 1px var(--fail); }
.case-title { min-width: 0; }
.case-id { font: 600 14px/1.5 var(--font-mono); overflow-wrap: anywhere; }
.case-desc { display: block; color: var(--muted); font-size: 13.5px; }
.case-headline { display: block; margin-top: 3px; color: var(--fail-text);
  font: 500 13px/1.5 var(--font-mono); overflow-wrap: anywhere; }
.case-meta { display: flex; align-items: center; gap: 10px; min-height: 24px; color: var(--muted);
  font-size: 13px; white-space: nowrap; font-variant-numeric: tabular-nums; }
.dots { display: inline-flex; gap: 4px; }
.dot { width: 9px; height: 9px; border-radius: 2px; background: var(--pass); }
.dot--fail, .dot--error { background: var(--fail); }
.chev { width: 18px; height: 18px; margin-top: 3px; color: var(--muted); transition: transform 0.18s ease; }
.case[open] .chev { transform: rotate(180deg); }
.case-body { padding: 4px 20px 20px; display: grid; gap: 16px; }
.label { margin: 0 0 8px; color: var(--muted); font: 600 11px/1.4 var(--font-mono); letter-spacing: 0.1em;
  text-transform: uppercase; }
.prompt { border-radius: 12px; background: var(--bg); border: 1px solid var(--line-soft); }
.prompt > summary { cursor: pointer; padding: 10px 14px; color: var(--muted); font-size: 13px; }
.prompt > summary:hover { color: var(--text); }
.prompt .msgs { padding: 0 14px 14px; display: grid; gap: 10px; }
.msg-role { color: var(--cyan); font: 600 11px/1.4 var(--font-mono); text-transform: uppercase;
  letter-spacing: 0.1em; }
.samples { display: grid; gap: 12px; }
.sample { border-radius: var(--radius-sm); background: var(--raised); border: 1px solid var(--line-soft);
  padding: 14px 16px; min-width: 0; }
.sample header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin-bottom: 10px; }
.sample header h3 { font-size: 15px; font-weight: 600; }
.sample header .lat { color: var(--muted); font-size: 12.5px; margin-left: auto; }
.checks { list-style: none; margin: 0 0 12px; padding: 0; display: grid; gap: 6px; }
.checks li { display: grid; grid-template-columns: 18px minmax(0, 1fr); gap: 8px; align-items: start;
  font-size: 13.5px; }
.checks .icon { font-weight: 700; line-height: 1.5; text-align: center; }
.checks .ok .icon { color: var(--pass); }
.checks .bad .icon { color: var(--fail); }
.checks code { font: 500 12.5px/1.5 var(--font-mono); overflow-wrap: anywhere; }
.checks .detail { display: block; color: var(--muted); font-size: 12.5px; overflow-wrap: anywhere; }
.checks .bad .detail { color: var(--fail-text); }
.warn { margin: 0 0 12px; padding: 8px 12px; border-radius: 10px; background: var(--warn-bg);
  color: var(--warn); font-size: 13px; }
.error-text { margin: 0; color: var(--fail-text); font: 500 13px/1.5 var(--font-mono);
  overflow-wrap: anywhere; }
pre { margin: 0; padding: 12px 14px; border-radius: 10px; background: var(--bg);
  border: 1px solid var(--line-soft); font: 12.5px/1.6 var(--font-mono); color: #cfe0e8;
  white-space: pre-wrap; overflow-wrap: anywhere; max-height: 340px; overflow: auto; }
.passing-samples > summary { cursor: pointer; color: var(--muted); font-size: 13px; padding: 4px 0; }
.passing-samples > summary:hover { color: var(--text); }
.passing-samples .samples { margin-top: 10px; }
footer { margin: 48px 0 32px; padding-top: 20px; border-top: 1px solid var(--line); color: var(--muted);
  font-size: 13px; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; }
@media (max-width: 860px) {
  .hero { grid-template-columns: 1fr; padding: 24px; }
  .stats { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 560px) {
  body { font-size: 14.5px; }
  .hero { padding: 22px 18px; }
  .verdict { padding: 14px 16px; gap: 14px; }
  .ring, .ring svg { width: 76px; height: 76px; }
  .stats { gap: 8px; }
  .stat { padding: 12px 14px; }
  .stat b { font-size: 23px; }
  .case > summary { grid-template-columns: auto minmax(0, 1fr) auto; padding: 14px 16px; }
  .case > summary .pill { grid-column: 1; grid-row: 1; }
  .case > summary .chev { grid-column: 3; grid-row: 1; }
  .case-title { grid-column: 1 / -1; grid-row: 2; }
  .case-meta { grid-column: 1 / -1; grid-row: 3; white-space: normal; flex-wrap: wrap; min-height: 0; }
  .case-body { padding: 0 14px 16px; }
  .sample { padding: 12px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important;
    scroll-behavior: auto !important; }
}
@media print {
  body { background: #fff; color: #111; }
  .tools { display: none; }
}
"""

SCRIPT = """
(function () {
  document.documentElement.classList.remove('no-js');
  var list = document.querySelector('.cases');
  var buttons = document.querySelectorAll('[data-filter-btn]');
  buttons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      list.setAttribute('data-filter', btn.getAttribute('data-filter-btn'));
      buttons.forEach(function (b) { b.setAttribute('aria-pressed', String(b === btn)); });
    });
  });
  var toggle = document.querySelector('[data-toggle-all]');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var cases = list.querySelectorAll('.case');
      var anyClosed = Array.prototype.some.call(cases, function (c) { return !c.open; });
      cases.forEach(function (c) { c.open = anyClosed; });
      toggle.textContent = anyClosed ? 'Collapse all' : 'Expand all';
    });
  }
})();
"""

CHEVRON = (
    '<svg class="chev" viewBox="0 0 20 20" aria-hidden="true" focusable="false">'
    '<path d="M5 7.5l5 5 5-5" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def _e(text: object) -> str:
    return escape(str(text), quote=True)


def _ring(rate: float) -> str:
    circumference = 2 * 3.14159265 * 40
    filled = circumference * max(0.0, min(rate, 1.0))
    return (
        f'<div class="ring" role="img" aria-label="{_e(percent(rate))} of cases passed">'
        '<svg viewBox="0 0 92 92" aria-hidden="true" focusable="false">'
        '<circle class="track" cx="46" cy="46" r="40" fill="none" stroke-width="9"/>'
        f'<circle class="value" cx="46" cy="46" r="40" fill="none" stroke-width="9" stroke-linecap="round" '
        f'stroke-dasharray="{filled:.2f} {circumference:.2f}"/></svg>'
        f"<span>{_e(percent(rate))}</span></div>"
    )


def _filter_button(key: str, text: str, count: int, pressed: bool) -> str:
    return (
        f'<button type="button" data-filter-btn="{key}" aria-pressed="{str(pressed).lower()}">'
        f'{text}<span class="count">{count}</span></button>'
    )


def _sample_dots(case: CaseResult) -> str:
    if len(case.samples) <= 1:
        return ""
    dots = "".join(
        f'<span class="dot dot--{s.status}" title="Sample {s.index + 1}: {STATUS_LABEL[s.status]}"></span>'
        for s in case.samples
    )
    return f'<span class="dots" aria-hidden="true">{dots}</span>'


def _prompt(case: CaseResult) -> str:
    blocks = []
    if case.case.system:
        blocks.append(
            f'<div><div class="msg-role">system</div><pre>{_e(case.case.system.strip())}</pre></div>'
        )
    for message in case.case.messages:
        blocks.append(
            f'<div><div class="msg-role">{_e(message["role"])}</div>'
            f"<pre>{_e(message['content'].strip())}</pre></div>"
        )
    turns = len(case.case.messages) + (1 if case.case.system else 0)
    return (
        '<details class="prompt"><summary>Prompt sent to the model '
        f"({turns} message{'s' if turns != 1 else ''})</summary>"
        f'<div class="msgs">{"".join(blocks)}</div></details>'
    )


def _sample(sample: SampleResult, total: int) -> str:
    title = f"Sample {sample.index + 1}" if total > 1 else "Result"
    parts = [
        '<article class="sample">',
        f'<header><span class="pill pill--{sample.status}">{STATUS_LABEL[sample.status]}</span>'
        f"<h3>{_e(title)}</h3>",
    ]
    if sample.latency_ms > 0 and sample.error is None:
        parts.append(f'<span class="lat">{_e(latency(sample.latency_ms))}</span>')
    parts.append("</header>")
    for warning in sample.warnings:
        parts.append(f'<p class="warn">{_e(warning)}</p>')
    if sample.error:
        parts.append(f'<p class="error-text">{_e(sample.error)}</p></article>')
        return "".join(parts)
    parts.append('<ul class="checks">')
    for result in sample.assertions:
        cls, icon, word = ("ok", "✓", "passed") if result.passed else ("bad", "✗", "failed")
        detail = f'<span class="detail">{_e(result.detail)}</span>' if result.detail else ""
        parts.append(
            f'<li class="{cls}"><span class="icon" aria-label="{word}">{icon}</span>'
            f"<span><code>{_e(result.label)}</code>{detail}</span></li>"
        )
    parts.append("</ul>")
    output = sample.output if sample.output else "(empty output)"
    parts.append(f'<div class="label">Output</div><pre>{_e(output)}</pre></article>')
    return "".join(parts)


def _case(case: CaseResult) -> str:
    status = case.status
    is_open = " open" if status != PASS else ""
    desc = f'<span class="case-desc">{_e(case.case.description)}</span>' if case.case.description else ""
    headline = case.headline()
    head = f'<span class="case-headline">{_e(headline)}</span>' if headline else ""
    if len(case.samples) > 1:
        counts = (
            f"{case.passed_samples}/{len(case.samples)} samples passed · {percent(case.pass_rate)} "
            f"(needs {percent(case.threshold)})"
        )
    else:
        counts = f"{case.passed_samples}/1 sample passed"
    timing = latency(case.latency_ms)
    if timing:
        counts += f" · {timing}"
    summary = (
        f'<summary><span class="pill pill--{status}">{STATUS_LABEL[status]}</span>'
        f'<span class="case-title"><span class="case-id">{_e(case.case.id)}</span>{desc}{head}</span>'
        f'<span class="case-meta">{_sample_dots(case)}<span>{_e(counts)}</span></span>{CHEVRON}</summary>'
    )
    failing = [s for s in case.samples if s.status != PASS]
    passing = [s for s in case.samples if s.status == PASS]
    body = [_prompt(case)]
    if failing:
        body.append(
            '<div class="samples">' + "".join(_sample(s, len(case.samples)) for s in failing) + "</div>"
        )
    if passing:
        rendered = (
            '<div class="samples">' + "".join(_sample(s, len(case.samples)) for s in passing) + "</div>"
        )
        if failing:
            noun = "sample" if len(passing) == 1 else "samples"
            body.append(
                f'<details class="passing-samples"><summary>Show {len(passing)} passing {noun}</summary>'
                f"{rendered}</details>"
            )
        else:
            body.append(rendered)
    return (
        f'<details class="case" data-status="{status}"{is_open}>{summary}'
        f'<div class="case-body">{"".join(body)}</div></details>'
    )


def render_html(run: RunResult, min_pass_rate: float = 1.0) -> str:
    ok = run.ok(min_pass_rate)
    failing = run.failed + run.errored
    if ok and failing == 0:
        verdict_title = f"All {run.total} cases passing"
    elif ok:
        verdict_title = f"Within threshold ({percent(min_pass_rate)})"
    else:
        verdict_title = f"{failing} case{'s' if failing != 1 else ''} failing"
    extras = [f"{run.errored} errored"] if run.errored else []
    extras += [f"{run.flaky} flaky"] if run.flaky else []
    verdict_sub = " · ".join([f"{run.passed} of {run.total} cases passed", *extras])
    started = run.started_at.strftime("%d %b %Y, %H:%M UTC")
    title = f"{run.suite.name} · promptcheck report"
    description = f"promptcheck results for {run.suite.name}: {run.passed} of {run.total} cases passed."

    facts = [
        ("Provider", f"<code>{_e(run.provider)}</code>"),
        ("Model", f"<code>{_e(run.model)}</code>" if run.model else "per case"),
        ("Started", _e(started)),
        ("Duration", _e(duration(run.duration_s))),
    ]
    facts_html = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in facts)
    stats = [
        ("pass", run.passed, "Passed"),
        ("fail" if run.failed else "", run.failed, "Failed"),
        ("err" if run.errored else "", run.errored, "Errors"),
        ("warn" if run.flaky else "", run.flaky, "Flaky"),
        ("cyan", run.sample_total, "Samples"),
    ]
    stats_html = "".join(
        f'<li class="stat{" stat--" + cls if cls else ""}"><b>{value}</b><span>{label}</span></li>'
        for cls, value, label in stats
    )
    notes = sorted(set(run.notices))
    notes_html = ""
    if notes:
        items = "".join(f"<li>{_e(n)}</li>" for n in notes)
        notes_html = f'<section class="notes" aria-label="Notes"><h2>Notes</h2><ul>{items}</ul></section>'
    failing_count = run.failed + run.errored
    filters = (
        '<div class="tools"><div class="filters" role="group" aria-label="Filter cases">'
        + _filter_button("all", "All", run.total, True)
        + _filter_button("failing", "Failing", failing_count, False)
        + _filter_button("passing", "Passing", run.passed, False)
        + '</div><button type="button" class="toggle" data-toggle-all>Expand all</button></div>'
    )
    cases_html = "".join(_case(c) for c in run.cases)
    suite_desc = f'<p class="desc">{_e(run.suite.description)}</p>' if run.suite.description else ""
    verdict_cls = "verdict--pass" if ok and failing == 0 else "verdict--fail"

    return f"""<!doctype html>
<html lang="en" class="no-js">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<meta name="description" content="{_e(description)}">
<meta name="generator" content="promptcheck {__version__}">
<meta name="theme-color" content="#04070a">
<meta property="og:type" content="website">
<meta property="og:title" content="{_e(title)}">
<meta property="og:description" content="{_e(description)}">
<link rel="icon" type="image/svg+xml" href="{FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header class="top">
<a class="brand" href="{REPO_URL}">{LOGO_SVG}<span>promptcheck</span></a>
<span class="tag">Test report · v{__version__}</span>
</header>
<main>
<section class="hero hud" aria-labelledby="suite-title">
<div>
<p class="eyebrow">// Suite<span class="path">{_e(run.suite.path.as_posix())}</span></p>
<h1 id="suite-title">{_e(run.suite.name)}</h1>
{suite_desc}
<dl class="facts">{facts_html}</dl>
</div>
<div class="verdict {verdict_cls}">{_ring(run.pass_rate)}
<div><strong>{_e(verdict_title)}</strong><small>{_e(verdict_sub)}</small></div>
</div>
</section>
<ul class="stats" aria-label="Totals">{stats_html}</ul>
{notes_html}
<section aria-labelledby="cases-title">
<div class="toolbar"><div><p class="kicker">// Results</p><h2 id="cases-title">Cases</h2></div>{filters}</div>
<div class="cases" data-filter="all">{cases_html}</div>
</section>
</main>
<footer>
<span>Generated by <a href="{REPO_URL}">promptcheck</a> {__version__} on {_e(started)}</span>
<span>{run.sample_total} samples · {_e(run.provider)}</span>
</footer>
</div>
<script>{SCRIPT}</script>
</body>
</html>
"""


def write_html(run: RunResult, path: str | Path, min_pass_rate: float = 1.0) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(run, min_pass_rate), encoding="utf-8")
    return target
