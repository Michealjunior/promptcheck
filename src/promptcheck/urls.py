"""Validation for API base URLs."""

from __future__ import annotations

from urllib.parse import urlsplit


def validate_base_url(value: object) -> str:
    """Return ``value`` without a trailing slash, or raise ValueError explaining the problem.

    A base URL must be an ``http`` or ``https`` URL with a host. An empty value is
    rejected rather than silently replaced by a provider default, because that would
    send the API key to a server the user did not choose.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be an http(s) URL with a host, got an empty value")
    if any(char.isspace() for char in value):
        raise ValueError(f"must not contain whitespace, got {value!r}")
    parsed = urlsplit(value)
    try:
        host, _port = parsed.hostname, parsed.port
    except ValueError:
        raise ValueError(f"has an invalid port: {value!r}") from None
    if parsed.scheme not in ("http", "https") or not host:
        raise ValueError(
            f"must be an http(s) URL with a host, such as https://api.example.com/v1; got {value!r}"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("must not contain credentials; put the key in an environment variable instead")
    if parsed.query or parsed.fragment:
        raise ValueError(f"must not contain a query string or fragment, got {value!r}")
    return value.rstrip("/")
