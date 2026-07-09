"""Session stores — subscribe to a run and persist it, one class per medium.

A store is the whole abstraction: `SessionStore` (abstract) owns the parts every backend
shares — subscribing to a Runner, projecting each event into a compact record, stamping
timestamps, and the meta/`started_at`/`ended_at` lifecycle. A concrete store implements just
three primitives — persist_meta / persist_record / flush — that decide WHERE bytes land:

  - FileSessionStore   a folder on disk, `<home>/sessions/<id>/` (meta.json + transcript.jsonl).
                       The default: durable, greppable, survives a crash line-by-line.
  - MemorySessionStore keeps records in a list (read `records` / `metadata`). Tests + ephemeral runs.
  - NullSessionStore   drops everything (used when recording is disabled).

Extend by subclassing SessionStore and implementing the three primitives — a database row,
object storage, an HTTP sink, etc. `open_session()` picks the default backend.

    store = open_session(chat.session, SessionMeta(model=model, provider=provider, cwd=cwd))
    store.attach(runner)               # subscribe
    store.user("summarize README.md")  # record a user turn before streaming
    ...
    await store.close()                # flush + stamp meta.ended_at

The projection is CURATED, not the raw firehose: streaming `message_update` deltas (one per
token) and structural no-ops are dropped; assistant turns, tool calls + results, and
effort/handoff/error markers are kept. Subagent activity is unwrapped and tagged with its
`depth`/`agent`. persist_record is synchronous and fire-and-forget (it runs inside an event
listener that must never block the run); only flush() may be async.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, TextIO

from pydantic import BaseModel

from curry_leaves.core.events import (
    AgentEvent,
    ApprovalEvent,
    CompactionEvent,
    ErrorEvent,
    HandoffEvent,
    MessageEnd,
    ThinkingEvent,
    ToolEnd,
    ToolStart,
    flatten,
)
from curry_leaves.util.paths import session_dir, session_meta_file, session_transcript_file


def _private_opener(path: str, flags: int) -> int:
    """open() opener that creates files 0600 — transcripts can contain anything the run saw."""
    return os.open(path, flags, 0o600)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    """Records embed pydantic models (content blocks, Usage, ...) straight from the event
    stream; dump those to plain JSON-able values instead of `json.dumps` choking on them.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return str(obj)


class SessionMeta(BaseModel):
    """Static session metadata (persisted by the store, e.g. as meta.json)."""

    model: str
    provider: str
    cwd: str


def _to_record(e: AgentEvent) -> dict[str, Any] | None:
    """Projection of an AgentEvent into a compact record, or None to skip it."""
    event, depth, name = flatten(e)
    tag: dict[str, Any] = {"depth": depth, "agent": name} if depth > 0 else {}
    if isinstance(event, MessageEnd):
        # Assistant turns only — user turns are recorded via user(); tool results arrive as tool_end.
        if event.message.role != "assistant":
            return None
        return {
            **tag,
            "kind": "assistant",
            "content": event.message.content,
            "stop_reason": event.message.stop_reason,
            "usage": event.message.usage,
        }
    if isinstance(event, ToolStart):
        return {**tag, "kind": "tool_start", "id": event.tool_call_id, "name": event.tool_name, "args": event.args}
    if isinstance(event, ToolEnd):
        return {
            **tag,
            "kind": "tool_end",
            "id": event.tool_call_id,
            "name": event.tool_name,
            "result": event.result.content,
            "is_error": event.result.is_error,
        }
    if isinstance(event, ThinkingEvent):
        return {**tag, "kind": "effort", "effort": event.effort}
    if isinstance(event, HandoffEvent):
        return {**tag, "kind": "handoff", "from": event.from_agent, "to": event.to_agent}
    if isinstance(event, ApprovalEvent):
        return {
            **tag,
            "kind": "approval",
            "tool": event.tool,
            "risk": event.risk,
            "granted": event.granted,
            "scope": event.scope,
        }
    if isinstance(event, CompactionEvent):
        return {
            **tag,
            "kind": "compaction",
            "reason": event.reason,
            "messages_before": event.messages_before,
            "messages_after": event.messages_after,
            "tokens_before": event.tokens_before,
            "summary": event.summary,
        }
    if isinstance(event, ErrorEvent):
        return {**tag, "kind": "error", "message": event.message, "fatal": event.fatal}
    # message_start / message_update (token deltas) / turn_start / turn_end / agent_end — noise.
    return None


class Subscribable(Protocol):
    """Anything a store can attach to: a Runner (or compatible) exposing `subscribe`."""

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]: ...


class SessionStore(ABC):
    """The shared base. Holds canonical `meta` and the attach/user/mark/close lifecycle; a
    subclass only says how to persist. Concrete stores call `persist_meta(self.metadata)` at the
    end of their constructor to stamp the session's start (the base never calls a primitive from
    its own constructor, so a subclass's resources are ready before any I/O).
    """

    def __init__(self, id: str, meta: SessionMeta) -> None:
        self.id = id
        self._meta: dict[str, Any] = {
            "session": id,
            **meta.model_dump(),
            "started_at": _now_iso(),
        }
        self._off: Callable[[], None] | None = None
        # Session-scoped tool approvals granted this run — stamped into meta at close (audit).
        self._session_approvals: set[str] = set()

    @property
    def metadata(self) -> dict[str, Any]:
        """Current metadata (canonical; updated in place with ended_at on close)."""
        return self._meta

    # ── primitives a backend implements ────────────────────────────────────────
    @abstractmethod
    def persist_meta(self, meta: dict[str, Any]) -> None:
        """Persist the metadata, replacing any prior value."""
        ...

    @abstractmethod
    def persist_record(self, record: dict[str, Any]) -> None:
        """Append one record to the transcript, preserving order. Must not throw."""
        ...

    @abstractmethod
    async def flush(self) -> None:
        """Flush buffers and release resources. Idempotent."""
        ...

    # ── shared recording lifecycle ─────────────────────────────────────────────
    def _write(self, record: dict[str, Any]) -> None:
        self.persist_record({"ts": _now_iso(), **record})

    def user(self, text: str) -> None:
        """Record a user turn (the event stream carries no user event, so frontends call this)."""
        self._write({"kind": "user", "text": text})

    def mark(self, kind: str, fields: dict[str, Any] | None = None) -> None:
        """Record a free-form control marker (e.g. a conversation reset)."""
        self._write({"kind": kind, **(fields or {})})

    def attach(self, runner: Subscribable) -> Callable[[], None]:
        """Subscribe to a runner's event stream so every event is recorded. Returns an unsubscribe
        fn; also stored so close() detaches automatically. Re-attaching (e.g. after a reset that
        builds a fresh runner) detaches the previous subscription first.
        """
        if self._off is not None:
            self._off()

        def on_event(e: AgentEvent) -> None:
            event, _depth, _name = flatten(e)
            if isinstance(event, ApprovalEvent) and event.granted and event.scope == "session":
                self._session_approvals.add(event.tool)
            record = _to_record(e)
            if record is not None:
                self._write(record)

        off = runner.subscribe(on_event)
        self._off = off
        return off

    async def close(self) -> None:
        """Detach, stamp meta.ended_at, and flush. Idempotent."""
        if self._off is not None:
            self._off()
        self._off = None
        self._meta["ended_at"] = _now_iso()
        if len(self._session_approvals) > 0:
            self._meta["approvals"] = list(self._session_approvals)
        self.persist_meta(self._meta)
        await self.flush()


class FileSessionStore(SessionStore):
    """The default backend: one directory per session under `<home>/sessions/<id>/`, holding a
    pretty-printed `meta.json` and an append-only `transcript.jsonl`. All I/O is best-effort —
    a disk hiccup disables further writes rather than breaking the run.
    """

    def __init__(self, id: str, meta: SessionMeta) -> None:
        super().__init__(id, meta)
        self.dir = session_dir(id)
        self.meta_path = session_meta_file(id)
        self.transcript_path = session_transcript_file(id)
        self._stream: TextIO | None = None
        try:
            # Transcripts replay everything the agent read/ran — owner-only, since the
            # default umask would leave them group/world-readable on shared machines.
            os.makedirs(self.dir, mode=0o700, exist_ok=True)
            os.chmod(self.dir, 0o700)
            self._stream = open(self.transcript_path, "a", encoding="utf-8", opener=_private_opener)
        except OSError:
            self._stream = None
        self.persist_meta(self.metadata)  # stamp meta.json at start

    def persist_meta(self, meta: dict[str, Any]) -> None:
        try:
            with open(self.meta_path, "w", encoding="utf-8", opener=_private_opener) as f:
                f.write(f"{json.dumps(meta, indent=2, default=_json_default)}\n")
        except OSError:
            pass  # best-effort

    def persist_record(self, record: dict[str, Any]) -> None:
        if self._stream is None:
            return
        try:
            self._stream.write(f"{json.dumps(record, default=_json_default)}\n")
            self._stream.flush()  # line-by-line durability is the whole point of the jsonl
        except OSError:
            pass  # best-effort

    async def flush(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        stream.close()


class MemorySessionStore(SessionStore):
    """Keeps everything in memory — read `records` and `metadata` directly. Nothing hits disk."""

    def __init__(self, id: str, meta: SessionMeta) -> None:
        super().__init__(id, meta)
        self.records: list[dict[str, Any]] = []

    def persist_meta(self, meta: dict[str, Any]) -> None:
        # metadata is canonical on the base; nothing to mirror.
        pass

    def persist_record(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    async def flush(self) -> None:
        pass


class NullSessionStore(SessionStore):
    """Discards all writes — the no-op backend used when recording is turned off."""

    def persist_meta(self, meta: dict[str, Any]) -> None:
        pass

    def persist_record(self, record: dict[str, Any]) -> None:
        pass

    async def flush(self) -> None:
        pass


def open_session(id: str, meta: SessionMeta) -> SessionStore:
    """Open the default store for a session: file-backed, or null when CURRY_LEAVES_NO_RECORD is set."""
    return NullSessionStore(id, meta) if os.environ.get("CURRY_LEAVES_NO_RECORD") else FileSessionStore(id, meta)
