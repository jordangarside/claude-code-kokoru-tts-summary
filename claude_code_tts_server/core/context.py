"""Request context and logging utilities."""

import uuid
from contextvars import ContextVar


def sanitize_for_log(text: str, max_len: int = 80) -> str:
    """Sanitize text for logging: replace newlines, truncate.

    Args:
        text: Text to sanitize.
        max_len: Maximum length before truncation.

    Returns:
        Sanitized text safe for single-line logging.
    """
    text = text.replace("\n", "\\n").replace("\r", "")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text

# Context variable for request ID
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Get the current request ID."""
    return request_id_var.get()


def set_request_id(request_id: str | None = None) -> str:
    """Set or generate a new request ID."""
    if request_id is None:
        request_id = uuid.uuid4().hex[:8]
    request_id_var.set(request_id)
    return request_id


def clear_request_id() -> None:
    """Clear the request ID."""
    request_id_var.set(None)
