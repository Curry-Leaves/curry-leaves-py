"""Server-Sent-Events parsing over an httpx streaming response body.

``iter_sse`` reads the streaming body, splits it into lines (``httpx``'s
``aiter_lines`` buffers partial lines across network chunks for us), keeps
only ``data:`` lines, strips the prefix, and yields each payload parsed as
JSON. An optional ``done_sentinel`` (OpenAI's ``[DONE]``) ends the stream.

Note it dispatches purely on the JSON payload — SSE ``event:`` lines and
blanks are simply skipped, which is exactly what both providers' parsers want.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from curry_leaves.util.retry import HttpError

_SKIP = object()
_DONE = object()


async def stream_json_sse(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    label: str,
    done_sentinel: str | None = None,
) -> AsyncIterator[Any]:
    """POST `body` to `url` and yield each SSE `data:` payload as parsed JSON. A response
    status >= 400 is raised as `HttpError` with a `<label> <code>: <body[:500]>` message —
    the one HTTP/error shape both providers share. Provider-specific request building and
    stream parsing stay at their own edges; only this glue is centralized here.
    """
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, headers=headers, json=body, timeout=None) as resp:
            if resp.status_code >= 400:
                text = ""
                try:
                    await resp.aread()
                    text = resp.text
                except Exception:
                    text = ""
                raise HttpError(resp.status_code, f"{label} {resp.status_code}: {text[:500]}")

            async for event in iter_sse(resp, done_sentinel=done_sentinel):
                yield event


async def iter_sse(
    response: httpx.Response,
    *,
    done_sentinel: str | None = None,
) -> AsyncIterator[Any]:
    async for line in response.aiter_lines():
        parsed = _handle_line(line, done_sentinel)
        if parsed is _DONE:
            return
        if parsed is not _SKIP:
            yield parsed


def _handle_line(line: str, done_sentinel: str | None) -> object:
    trimmed = line[:-1] if line.endswith("\r") else line
    if not trimmed.startswith("data:"):
        return _SKIP
    payload = trimmed[5:].strip()
    if not payload:
        return _SKIP
    if done_sentinel is not None and payload == done_sentinel:
        return _DONE
    return json.loads(payload)
