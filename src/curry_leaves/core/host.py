"""The Host — the frontend seam. The kernel emits events and (rarely) requests input
through this one interface, so the same engine drives a CLI, a test harness, or an
SDK caller without knowing which.

Two methods:
  - emit(event)      fire-and-forget; observers render progress. Never blocks the run.
  - request(req)     ask the user something and await a typed answer (the `ask` tool).

The default host is headless: it drops events on the floor and answers every request
with the request's default, so a run never hangs waiting for a human that isn't there.
"""

from __future__ import annotations

from typing import Callable, Generic, Literal, Protocol, TypeVar

from curry_leaves.core.events import AgentEvent, ev

T = TypeVar("T")


class Request(Generic[T]):
    """Base for an engine->host request. Each concrete request carries its inputs plus a
    `default: T` — the value the host returns when it can't or won't answer (no frontend,
    headless run). One bidirectional seam carries them all; add a capability by adding a
    Request kind, never a method.
    """

    kind: str
    default: T


class AskUser(Request[str]):
    """Put a question to the user and wait for their answer (the `ask` tool)."""

    kind: Literal["ask_user"] = "ask_user"

    def __init__(self, question: str, options: list[str], default: str) -> None:
        self.question = question
        self.options = options
        self.default = default


# How far a tool approval persists. `deny` = refused.
ApprovalChoice = Literal["deny", "once", "session", "always"]


class ApproveTool(Request[ApprovalChoice]):
    """Ask the user to approve a tool call the permission policy flagged for prompting."""

    kind: Literal["approve_tool"] = "approve_tool"

    def __init__(
        self,
        tool: str,
        args: dict[str, object],
        risk: str,
        reason: str,
        default: ApprovalChoice,
    ) -> None:
        self.tool = tool
        self.args = args
        self.risk = risk
        self.reason = reason
        self.default = default


class Host(Protocol):
    def emit(self, event: AgentEvent) -> None: ...

    async def request(self, req: Request[T]) -> T: ...


class DefaultHost:
    """Headless host: no UI. Emits are observable via listeners; requests get the default."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[AgentEvent], None]] = []

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        self._listeners.append(fn)

        def unsubscribe() -> None:
            if fn in self._listeners:
                self._listeners.remove(fn)

        return unsubscribe

    def emit(self, event: AgentEvent) -> None:
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:
                pass  # an observer must never break the run

    async def request(self, req: Request[T]) -> T:
        return req.default


class SubagentHost:
    """Wraps a parent host for a subagent: the child's events are re-emitted to the parent
    as SubagentActivity (tagged with depth), and requests pass through to the parent so a
    subagent can still ask the user. Mirrors how the Python SubagentHost nests events.
    """

    def __init__(self, parent: Host, depth: int, name: str) -> None:
        self._parent = parent
        self._depth = depth
        self._name = name

    def emit(self, event: AgentEvent) -> None:
        self._parent.emit(ev.subagent(event, self._depth, self._name))

    async def request(self, req: Request[T]) -> T:
        return await self._parent.request(req)
