"""Events the loop yields and frontends consume.

Design rule: the loop PRODUCES events; consumers (a CLI, an SDK caller) CONSUME
them. That separation is what lets one engine drive many UIs.

Two tiers, following the pattern that keeps the surface small:
  1. STRUCTURAL lifecycle — message / tool / turn / run boundaries. Small, stable.
  2. STREAMING deltas     — text / thinking / tool-arg chunks. Carried as a `delta`
     PAYLOAD on MessageUpdate, not as many event classes.

A simple consumer ignores `delta` and re-renders from the message snapshot; an
advanced consumer (append-only token rendering, live tool-arg previews) reads it.

Events are in-memory only (never persisted), so they're plain models.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from curry_leaves.core.messages import AssistantMessage, Message, ToolResultMessage

# ── tier 2: the streaming delta (a payload, not a top-level event) ───────────


class Delta(BaseModel):
    """What changed since the previous MessageUpdate.

    ``kind`` says which stream advanced; ``block_index`` says which content
    block it belongs to; ``value`` is the incremental chunk — a text/thinking
    fragment, a signature fragment, or a slice of partial JSON for a tool
    call's arguments.
    """

    kind: Literal["text", "thinking", "signature", "tool_args"]
    block_index: int
    value: str


# ── tier 1: structural lifecycle events ──────────────────────────────────────


class MessageStart(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: Message


class MessageUpdate(BaseModel):
    type: Literal["message_update"] = "message_update"
    message: AssistantMessage
    delta: Delta | None = None


class MessageEnd(BaseModel):
    type: Literal["message_end"] = "message_end"
    message: Message


class ToolStart(BaseModel):
    type: Literal["tool_start"] = "tool_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, object] = Field(default_factory=dict)


class ToolEnd(BaseModel):
    type: Literal["tool_end"] = "tool_end"
    tool_call_id: str
    tool_name: str
    result: ToolResultMessage


class TurnStart(BaseModel):
    """A new model call is about to begin (paired with TurnEnd for UI grouping)."""

    type: Literal["turn_start"] = "turn_start"


class TurnEnd(BaseModel):
    """One model call + its tools finished."""

    type: Literal["turn_end"] = "turn_end"
    message: AssistantMessage
    tool_results: list[ToolResultMessage] = Field(default_factory=list)


class ErrorEvent(BaseModel):
    """A surfaced error.

    ``fatal`` means the run is ending; otherwise it's advisory (e.g. a
    retryable provider hiccup the loop is handling).
    """

    type: Literal["error"] = "error"
    message: str
    fatal: bool = False


class ThinkingEvent(BaseModel):
    """Auto-thinking classified this turn's difficulty and chose a reasoning effort."""

    type: Literal["thinking"] = "thinking"
    effort: str


class HandoffEvent(BaseModel):
    """Control was transferred to another agent.

    The conversation continues under ``to_agent``, which now drives the same
    transcript (the prior agent does not resume).
    """

    type: Literal["handoff"] = "handoff"
    from_agent: str
    to_agent: str


class CompactionEvent(BaseModel):
    """The conversation history was compacted to fit the context window.

    Older messages replaced by a model-generated ``summary``. ``reason`` is
    why it fired; the counts/tokens describe the before/after so a UI or the
    session store can surface it.
    """

    type: Literal["compaction"] = "compaction"
    reason: Literal["auto", "manual"]
    messages_before: int
    messages_after: int
    tokens_before: int
    summary: str


class ElisionEvent(BaseModel):
    """Stale tool results were stubbed out to reclaim context (an elision sweep).

    Originals are preserved in the blob store behind ``artifact://`` stubs, so
    nothing is lost — the counts let a UI or the session store surface how much
    of the window was reclaimed.
    """

    type: Literal["elision"] = "elision"
    results_elided: int
    tokens_before: int
    tokens_reclaimed: int


class ApprovalEvent(BaseModel):
    """The permission engine resolved a tool call that required a user decision.

    ``granted`` is the outcome; ``scope`` says how far a grant persists —
    ``once`` (this call), ``session`` (recorded in session meta), ``always``
    (written to settings.json), or ``deny``. Auto-allowed calls do NOT emit
    this (their tool_start/tool_end already show them).
    """

    type: Literal["approval"] = "approval"
    tool: str
    risk: str
    granted: bool
    scope: Literal["once", "session", "always", "deny", "auto"]


class AgentEnd(BaseModel):
    """The run is over: no pending tools, no queued follow-ups."""

    type: Literal["agent_end"] = "agent_end"


class SubagentActivity(BaseModel):
    """A subagent's event, surfaced to the parent wrapped with which agent produced it.

    ``depth`` (1 = direct child) and ``name`` tag the source. The inner
    ``event`` is the FULL agent event — a subagent's ``tool_start``,
    ``message_update``, etc. — so a consumer sees the complete event set from
    subagents, just tagged with their source.
    """

    type: Literal["subagent_activity"] = "subagent_activity"
    event: "AgentEvent"
    depth: int
    name: str


AgentEvent = Annotated[
    Union[
        MessageStart,
        MessageUpdate,
        MessageEnd,
        ToolStart,
        ToolEnd,
        TurnStart,
        TurnEnd,
        ErrorEvent,
        ThinkingEvent,
        HandoffEvent,
        CompactionEvent,
        ElisionEvent,
        ApprovalEvent,
        AgentEnd,
        SubagentActivity,
    ],
    Field(discriminator="type"),
]

SubagentActivity.model_rebuild()


# ── constructors — keep call sites readable ──────────────────────────────────


class ev:
    """Namespace of tiny constructors, mirroring the TS `ev` object."""

    @staticmethod
    def message_start(message: Message) -> MessageStart:
        return MessageStart(message=message)

    @staticmethod
    def message_update(message: AssistantMessage, delta: Delta | None) -> MessageUpdate:
        return MessageUpdate(message=message, delta=delta)

    @staticmethod
    def message_end(message: Message) -> MessageEnd:
        return MessageEnd(message=message)

    @staticmethod
    def tool_start(tool_call_id: str, tool_name: str, args: dict[str, object]) -> ToolStart:
        return ToolStart(tool_call_id=tool_call_id, tool_name=tool_name, args=args)

    @staticmethod
    def tool_end(tool_call_id: str, tool_name: str, result: ToolResultMessage) -> ToolEnd:
        return ToolEnd(tool_call_id=tool_call_id, tool_name=tool_name, result=result)

    @staticmethod
    def turn_start() -> TurnStart:
        return TurnStart()

    @staticmethod
    def turn_end(message: AssistantMessage, tool_results: list[ToolResultMessage]) -> TurnEnd:
        return TurnEnd(message=message, tool_results=tool_results)

    @staticmethod
    def error(message: str, fatal: bool = False) -> ErrorEvent:
        return ErrorEvent(message=message, fatal=fatal)

    @staticmethod
    def thinking(effort: str) -> ThinkingEvent:
        return ThinkingEvent(effort=effort)

    @staticmethod
    def handoff(from_agent: str, to_agent: str) -> HandoffEvent:
        return HandoffEvent(from_agent=from_agent, to_agent=to_agent)

    @staticmethod
    def compaction(
        reason: Literal["auto", "manual"],
        messages_before: int,
        messages_after: int,
        tokens_before: int,
        summary: str,
    ) -> CompactionEvent:
        return CompactionEvent(
            reason=reason,
            messages_before=messages_before,
            messages_after=messages_after,
            tokens_before=tokens_before,
            summary=summary,
        )

    @staticmethod
    def elision(results_elided: int, tokens_before: int, tokens_reclaimed: int) -> ElisionEvent:
        return ElisionEvent(
            results_elided=results_elided,
            tokens_before=tokens_before,
            tokens_reclaimed=tokens_reclaimed,
        )

    @staticmethod
    def approval(
        tool: str,
        risk: str,
        granted: bool,
        scope: Literal["once", "session", "always", "deny", "auto"],
    ) -> ApprovalEvent:
        return ApprovalEvent(tool=tool, risk=risk, granted=granted, scope=scope)

    @staticmethod
    def agent_end() -> AgentEnd:
        return AgentEnd()

    @staticmethod
    def subagent(event: "AgentEvent", depth: int, name: str) -> SubagentActivity:
        return SubagentActivity(event=event, depth=depth, name=name)


def flatten(e: "AgentEvent") -> tuple[object, int, str]:
    """Normalize any event into (event, depth, name).

    Parent events -> depth 0, name ""; subagent activity is unwrapped to its
    inner event with its depth/name.
    """
    if isinstance(e, SubagentActivity):
        return e.event, e.depth, e.name
    return e, 0, ""
