"""A tiny content-addressed blob store — where oversized tool output is offloaded.

No single tool result should dominate the context window (a 750 KB fetched page or a
runaway `bash` dump can blow the request past the model's limit). So oversized output
is stored WHOLE here and the model keeps a head+tail preview plus an `artifact://<id>`
URL it can `read` (with offset/limit) for the rest.

This in-memory store lives for the duration of a run tree (shared across subagents).
"""

from __future__ import annotations

import hashlib
from typing import Callable


def truncate_with_blob(
    text: str,
    limit: int,
    blobs: "BlobStore | None",
    *,
    stored: Callable[[str, int], str],
    dropped: str = "... [truncated]",
) -> str:
    """Cap `text` at `limit` chars for the model. Under the limit it's returned unchanged;
    over it, the WHOLE text is offloaded to `blobs` (if present) and the caller keeps a
    head preview plus a note. `stored(blob_id, total_len)` builds the note when the full
    text was saved (so it can cite `artifact://<id>`); `dropped` is the fallback note when
    there's no blob store. Both are appended after the preview on their own line.
    """
    if len(text) <= limit:
        return text
    preview = text[:limit]
    if blobs is not None:
        note = stored(blobs.put_text(text), len(text))
    else:
        note = dropped
    return f"{preview}\n{note}"


class BlobStore:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def put_text(self, text: str) -> str:
        """Store text, returning a stable content-addressed id (sha256[:16])."""
        id_ = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        self._store[id_] = text
        return id_

    def get_text(self, id_: str) -> str | None:
        """Full text for an id, or None if unknown."""
        return self._store.get(id_)

    def get(
        self, id_: str, offset: int | None = None, limit: int | None = None
    ) -> str | None:
        """A slice of a stored blob by 1-based line offset and max line count — what `read`
        uses to page through an offloaded artifact.
        """
        text = self._store.get(id_)
        if text is None:
            return None
        if offset is None and limit is None:
            return text
        lines = text.split("\n")
        start = max(0, (offset if offset is not None else 1) - 1)
        end = start + limit if limit is not None else len(lines)
        return "\n".join(lines[start:end])
