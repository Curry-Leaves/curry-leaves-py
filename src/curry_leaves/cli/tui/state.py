"""TUI state — turn the loop's event stream into a renderable transcript.

The event stream (tier-1 lifecycle + tier-2 deltas) is folded into an ordered list
of `Part`s per assistant turn. Finished turns become `Entry`s that get committed to
scrollback (the Textual transcript log widget appends them once, like Ink's
<Static>); the in-progress turn lives in `live` and re-renders as tokens stream in.
A single reducer handles both the main stream and subagent activity, so the widget
layer just dispatches events and renders state.

Kept as plain dataclasses + a pure `reduce()` function — no Textual dependency here —
mirroring the TS state.ts, which is pure data transformation independent of React.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Literal, Optional, Union

from curry_leaves.core.events import AgentEvent

# ── the pieces of one assistant turn ─────────────────────────────────────────


@dataclass(frozen=True)
class TextPart:
    text: str
    kind: Literal["text"] = "text"


@dataclass(frozen=True)
class ThinkingPart:
    text: str
    depth: int = 0  # >0 when this is a subagent's reasoning
    agent: Optional[str] = None  # the subagent that produced it (for nested reasoning)
    kind: Literal["thinking"] = "thinking"


@dataclass(frozen=True)
class ToolPart:
    id: str
    name: str
    args: dict[str, object]
    status: Literal["running", "ok", "error"]
    depth: int  # 0 = main agent, >0 = subagent nesting
    result: Optional[str] = None
    agent: Optional[str] = None  # the subagent that ran this tool (None for the main agent)
    kind: Literal["tool"] = "tool"


@dataclass(frozen=True)
class EffortPart:
    effort: str
    kind: Literal["effort"] = "effort"


@dataclass(frozen=True)
class HandoffPart:
    from_: str
    to: str
    kind: Literal["handoff"] = "handoff"


@dataclass(frozen=True)
class ErrorPart:
    message: str
    kind: Literal["error"] = "error"


@dataclass(frozen=True)
class CompactionPart:
    reason: Literal["auto", "manual"]
    messages_before: int
    messages_after: int
    kind: Literal["compaction"] = "compaction"


@dataclass(frozen=True)
class StatsPart:
    """A per-turn footer: wall-clock time + tokens (and cost when non-local)."""

    seconds: float
    in_tok: int
    out_tok: int
    cost: float
    kind: Literal["stats"] = "stats"


Part = Union[
    TextPart,
    ThinkingPart,
    ToolPart,
    EffortPart,
    HandoffPart,
    ErrorPart,
    CompactionPart,
    StatsPart,
]

# ── a transcript entry (one committed row of history) ────────────────────────


@dataclass(frozen=True)
class BannerData:
    model: str
    provider: str
    cwd: str
    session: str
    version: str
    tool_groups: list[tuple[str, list[str]]]
    skills: list[str]
    tool_count: int
    skill_count: int
    subagents: str


@dataclass(frozen=True)
class UserEntry:
    id: int
    text: str
    role: Literal["user"] = "user"


@dataclass(frozen=True)
class AssistantEntry:
    id: int
    parts: list[Part]
    role: Literal["assistant"] = "assistant"


@dataclass(frozen=True)
class NoticeEntry:
    id: int
    lines: list[str]
    role: Literal["notice"] = "notice"


@dataclass(frozen=True)
class BannerEntry:
    id: int
    data: BannerData
    role: Literal["banner"] = "banner"


Entry = Union[UserEntry, AssistantEntry, NoticeEntry, BannerEntry]

Status = Literal["idle", "thinking", "working"]


@dataclass
class State:
    entries: list[Entry] = field(default_factory=list)
    live: Optional[list[Part]] = None  # the in-progress assistant turn, or None when idle
    status: Status = "idle"
    next_id: int = 0


def initial_state() -> State:
    return State(entries=[], live=None, status="idle", next_id=0)


# ── actions the reducer accepts ───────────────────────────────────────────────


@dataclass(frozen=True)
class UserAction:
    text: str
    type: Literal["user"] = "user"


@dataclass(frozen=True)
class NoticeAction:
    lines: list[str]
    type: Literal["notice"] = "notice"


@dataclass(frozen=True)
class EventAction:
    """From the main stream."""

    e: AgentEvent
    type: Literal["event"] = "event"


@dataclass(frozen=True)
class SubAction:
    """From a subagent."""

    e: AgentEvent
    depth: int
    name: str
    type: Literal["sub"] = "sub"


@dataclass(frozen=True)
class StatsAction:
    """Per-turn footer."""

    seconds: float
    in_tok: int
    out_tok: int
    cost: float
    type: Literal["stats"] = "stats"


@dataclass(frozen=True)
class FinalizeAction:
    """Commit the live turn to history."""

    type: Literal["finalize"] = "finalize"


@dataclass(frozen=True)
class ClearAction:
    """Wipe scrollback."""

    type: Literal["clear"] = "clear"


Action = Union[UserAction, NoticeAction, EventAction, SubAction, StatsAction, FinalizeAction, ClearAction]

# ── helpers ──────────────────────────────────────────────────────────────────

MAX_RESULT = 200


def snippet(text: str, max_len: int = MAX_RESULT) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    return f"{one_line[:max_len]}…" if len(one_line) > max_len else one_line


def _result_text(result: object) -> str:
    content = getattr(result, "content", None)
    if not content:
        return ""
    first = content[0]
    text = getattr(first, "text", None)
    if getattr(first, "type", None) == "text" and text:
        return str(text)
    return ""


def _append_to(parts: list[Part], kind: Literal["text", "thinking"], value: str) -> list[Part]:
    """Append streamed `value` to the trailing part if it matches `kind`, else start a new one."""
    if parts:
        last = parts[-1]
        if kind == "text" and isinstance(last, TextPart):
            return [*parts[:-1], replace(last, text=last.text + value)]
        if kind == "thinking" and isinstance(last, ThinkingPart):
            return [*parts[:-1], replace(last, text=last.text + value)]
    if kind == "text":
        return [*parts, TextPart(text=value)]
    return [*parts, ThinkingPart(text=value)]


def _append_sub_thinking(parts: list[Part], value: str, depth: int, agent: str) -> list[Part]:
    """Append a subagent's reasoning/output, keyed by agent+depth so streams don't merge."""
    if parts:
        last = parts[-1]
        if isinstance(last, ThinkingPart) and last.agent == agent and last.depth == depth:
            return [*parts[:-1], replace(last, text=last.text + value)]
    return [*parts, ThinkingPart(text=value, depth=depth, agent=agent)]


def _fold_event(parts: list[Part], e: AgentEvent, depth: int, name: str = "") -> list[Part]:
    """Fold one loop event (main or subagent) into the live parts list."""
    if e.type == "message_update":
        d = e.delta
        if d is None:
            return parts
        if depth == 0:
            if d.kind == "text":
                return _append_to(parts, "text", d.value)
            if d.kind == "thinking":
                return _append_to(parts, "thinking", d.value)
            return parts
        # Subagent prose (its reasoning and streamed output) → dim, indented, agent-labeled.
        if d.kind in ("text", "thinking"):
            return _append_sub_thinking(parts, d.value, depth, name)
        return parts
    if e.type == "tool_start":
        return [
            *parts,
            ToolPart(
                id=e.tool_call_id,
                name=e.tool_name,
                args=e.args,
                status="running",
                depth=depth,
                agent=name if depth > 0 else None,
            ),
        ]
    if e.type == "tool_end":
        text = _result_text(e.result)
        out: list[Part] = []
        for p in parts:
            if isinstance(p, ToolPart) and p.id == e.tool_call_id:
                out.append(
                    replace(
                        p,
                        status="error" if e.result.is_error else "ok",
                        result=snippet(text),
                    )
                )
            else:
                out.append(p)
        return out
    if e.type == "thinking":
        return [*parts, EffortPart(effort=e.effort)] if depth == 0 else parts
    if e.type == "handoff":
        return [*parts, HandoffPart(from_=e.from_agent, to=e.to_agent)]
    if e.type == "compaction":
        return [
            *parts,
            CompactionPart(
                reason=e.reason,
                messages_before=e.messages_before,
                messages_after=e.messages_after,
            ),
        ]
    if e.type == "error":
        return [*parts, ErrorPart(message=e.message)]
    return parts


def _status_for(parts: list[Part]) -> Status:
    """Derive a coarse status from the live parts (drives the status bar / spinner)."""
    last = parts[-1] if parts else None
    if isinstance(last, ToolPart) and last.status == "running":
        return "working"
    if any(isinstance(p, ToolPart) and p.status == "running" for p in parts):
        return "working"
    if isinstance(last, (ThinkingPart, EffortPart)):
        return "thinking"
    return "working"


def reduce(state: State, action: Action) -> State:
    if isinstance(action, UserAction):
        return State(
            entries=[*state.entries, UserEntry(id=state.next_id, text=action.text)],
            live=[],  # open a fresh assistant turn
            status="thinking",
            next_id=state.next_id + 1,
        )
    if isinstance(action, NoticeAction):
        return State(
            entries=[*state.entries, NoticeEntry(id=state.next_id, lines=action.lines)],
            live=state.live,
            status=state.status,
            next_id=state.next_id + 1,
        )
    if isinstance(action, EventAction):
        if state.live is None:
            return state
        live = _fold_event(state.live, action.e, 0)
        return State(entries=state.entries, live=live, status=_status_for(live), next_id=state.next_id)
    if isinstance(action, SubAction):
        if state.live is None:
            return state
        live = _fold_event(state.live, action.e, action.depth, action.name)
        return State(entries=state.entries, live=live, status=_status_for(live), next_id=state.next_id)
    if isinstance(action, StatsAction):
        if state.live is None:
            return state
        live = [
            *state.live,
            StatsPart(seconds=action.seconds, in_tok=action.in_tok, out_tok=action.out_tok, cost=action.cost),
        ]
        return State(entries=state.entries, live=live, status=state.status, next_id=state.next_id)
    if isinstance(action, FinalizeAction):
        if state.live is None or len(state.live) == 0:
            return State(entries=state.entries, live=None, status="idle", next_id=state.next_id)
        return State(
            entries=[*state.entries, AssistantEntry(id=state.next_id, parts=state.live)],
            live=None,
            status="idle",
            next_id=state.next_id + 1,
        )
    if isinstance(action, ClearAction):
        return State(entries=[], live=None, status="idle", next_id=state.next_id)
    return state


def compact_args(args: dict[str, object], max_len: int = 60) -> str:
    """Compact a tool's args to a single line for display."""
    s = json.dumps(args)
    return f"{s[:max_len]}…" if len(s) > max_len else s
