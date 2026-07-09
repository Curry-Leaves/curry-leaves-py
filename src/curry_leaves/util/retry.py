"""RetryPolicy — retry TRANSIENT provider failures with backoff.

A 503, a 429 rate-limit, or a dropped connection is bad luck, not a bug — retrying
usually works. This policy decides WHAT is worth retrying and HOW LONG to wait. It's
injected into the loop, so the loop never learns about HTTP.
"""

from typing import Protocol

import httpx


class HttpError(Exception):
    """Raised by providers on a non-2xx response, carrying the status for retry decisions."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class RetryPolicy(Protocol):
    max_attempts: int

    def is_transient(self, e: BaseException) -> bool: ...

    def delay(self, attempt: int) -> float:
        """Seconds to wait before the given (1-based) retry attempt."""
        ...


# Status codes worth retrying: timeouts, rate limits, and 5xx server hiccups.
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class DefaultRetryPolicy:
    def __init__(
        self,
        max_attempts: int = 3,  # retries AFTER the first try
        base_delay: float = 0.5,
        max_delay: float = 8.0,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    def is_transient(self, e: BaseException) -> bool:
        if isinstance(e, HttpError):
            return e.status in _TRANSIENT_STATUS
        # httpx raises a TransportError (e.g. ConnectError) on network-level failures
        # (DNS, reset, refused) — the Python equivalent of fetch's TypeError.
        if isinstance(e, httpx.TransportError):
            return True
        return False

    def delay(self, attempt: int) -> float:
        return float(min(self.max_delay, self.base_delay * 2 ** (attempt - 1)))
