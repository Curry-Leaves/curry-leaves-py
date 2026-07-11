"""Session forking — read a recorded transcript back into live `Message`s, so a new session
can pick up mid-conversation as its own branch.

`store.py` only ever writes a transcript forward; this module is the read path, used solely
to seed a fork. It is deliberately narrow — not a general "load a session" API — and only
understands the curated projection `_to_record` produces:

    user       -> UserMessage
    assistant  -> AssistantMessage (its `content` already carries any tool_call blocks)
    tool_end   -> ToolResultMessage (tool_start is redundant: the assistant's ToolCallBlock
                  already has id/name/args, so only the result needs replaying)

Root-level records only (`depth` absent/0) — subagent activity is tagged and skipped, since
it was never part of the root Runner's `messages`.

    records = load_transcript(source_id)
    offsets = user_turn_offsets(records)          # pick a fork point: after which user turn?
    new_id  = fork_session(source_id, upto_turn=2, meta=SessionMeta(...))
"""

from __future__ import annotations

import json
from typing import Any, Optional

from curry_leaves.core.messages import (
    AssistantMessage,
    Message,
    TextBlock,
    ToolResultMessage,
    UserMessage,
)
from curry_leaves.session.store import FileSessionStore, SessionMeta
from curry_leaves.util.paths import session_meta_file, session_transcript_file


def load_transcript(id: str) -> list[dict[str, Any]]:
    """Read a session's transcript.jsonl back into records, in order. Empty if the session
    has no transcript (e.g. it was recorded with CURRY_LEAVES_NO_RECORD, or never existed).
    """
    path = session_transcript_file(id)
    records: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    except OSError:
        return []
    return records


def load_meta(id: str) -> Optional[dict[str, Any]]:
    """Read a session's meta.json, or None if it doesn't exist / isn't readable."""
    path = session_meta_file(id)
    try:
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data
    except (OSError, json.JSONDecodeError):
        return None


def user_turn_offsets(records: list[dict[str, Any]]) -> list[int]:
    """Record indices where each root-level user turn starts — the valid fork points.
    `upto_turn=N` in fork_session keeps records[: offsets[N]] (everything through the Nth
    user turn's reply, before the (N+1)th user turn begins).
    """
    return [i for i, r in enumerate(records) if r.get("kind") == "user" and "depth" not in r]


def _is_root(record: dict[str, Any]) -> bool:
    return "depth" not in record  # tagged records (depth>0) are subagent activity


def transcript_to_messages(records: list[dict[str, Any]]) -> list[Message]:
    """Replay root-level transcript records into the live Message list a Runner expects."""
    messages: list[Message] = []
    for r in records:
        if not _is_root(r):
            continue
        kind = r.get("kind")
        if kind == "user":
            if "content" in r:  # multimodal turn — recorded blocks replay verbatim
                messages.append(UserMessage.model_validate({"content": r["content"]}))
            else:
                messages.append(UserMessage(content=[TextBlock(text=r.get("text", ""))]))
        elif kind == "assistant":
            messages.append(
                AssistantMessage.model_validate(
                    {
                        "content": r.get("content", []),
                        "stop_reason": r.get("stop_reason"),
                        "usage": r.get("usage"),
                    }
                )
            )
        elif kind == "tool_end":
            messages.append(
                ToolResultMessage(
                    tool_call_id=r["id"],
                    tool_name=r["name"],
                    content=[TextBlock(text=str(r.get("result", "")))],
                    is_error=bool(r.get("is_error", False)),
                )
            )
        # tool_start / effort / handoff / approval / compaction / error — not part of message state.
    return messages


def fork_session(
    source_id: str, new_id: str, meta: SessionMeta, *, upto_turn: Optional[int] = None
) -> tuple[FileSessionStore, list[Message]]:
    """Copy `source_id`'s transcript (optionally truncated after the `upto_turn`-th user turn,
    0-indexed; None means the whole thing) into a brand-new session `new_id`, stamping the new
    session's meta with `forked_from`. Returns the new (open) store plus the replayed Messages,
    so the caller can hand both straight to a Runner/Chat — mirroring open_session(), the store
    is left open for the caller to keep recording into and eventually close(). Best-effort like
    the rest of the store layer — an unreadable source just forks empty.
    """
    records = load_transcript(source_id)
    if upto_turn is not None:
        offsets = user_turn_offsets(records)
        if upto_turn < len(offsets):
            end = offsets[upto_turn + 1] if upto_turn + 1 < len(offsets) else len(records)
            records = records[:end]

    store = FileSessionStore(new_id, meta)
    store.metadata["forked_from"] = {"session": source_id, "upto_turn": upto_turn}
    store.persist_meta(store.metadata)
    for r in records:
        store.persist_record(r)  # each write flushes (jsonl durability), per FileSessionStore

    return store, transcript_to_messages(records)
