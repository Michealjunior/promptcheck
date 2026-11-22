"""Report writers: terminal, JUnit XML, Markdown and HTML."""

from promptcheck.reporters.html import render_html, write_html
from promptcheck.reporters.junit import render_junit, write_junit
from promptcheck.reporters.markdown import render_markdown, write_markdown
from promptcheck.reporters.terminal import TerminalReporter, use_color

__all__ = [
    "TerminalReporter",
    "render_html",
    "render_junit",
    "render_markdown",
    "use_color",
    "write_html",
    "write_junit",
    "write_markdown",
]
