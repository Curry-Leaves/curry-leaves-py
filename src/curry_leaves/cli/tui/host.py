"""TuiHost — the full-screen UI's interactive host adapter.

Extends DefaultHost (keeps event emit/subscribe) and bridges `request` into Textual: a
request is stored as `current` and a change-listener fires so the App re-renders (pushes
an ask/approve prompt screen). When the user answers, the App calls `respond(value)`,
which resolves the awaiting tool/engine via an `asyncio.Future` (the Python analogue of
the TS `Promise` the request awaits). Only `ask_user` / `approve_tool` prompt; anything
else returns its default (headless-safe).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar, Union

from curry_leaves.core.host import ApproveTool, AskUser, DefaultHost, Request

T = TypeVar("T")


@dataclass
class PendingRequest:
    req: Union[AskUser, ApproveTool]
    respond: Callable[[object], None]


class TuiHost(DefaultHost):
    def __init__(self) -> None:
        super().__init__()
        self._pending: Optional[PendingRequest] = None
        self._listener: Optional[Callable[[], None]] = None

    def on_change(self, fn: Callable[[], None]) -> Callable[[], None]:
        """The App registers here; called whenever the pending request appears or clears."""
        self._listener = fn

        def unsubscribe() -> None:
            if self._listener is fn:
                self._listener = None

        return unsubscribe

    @property
    def current(self) -> Optional[PendingRequest]:
        """The request awaiting an answer, or None. Read by the App to decide what to render."""
        return self._pending

    async def request(self, req: Request[T]) -> T:
        if not isinstance(req, (AskUser, ApproveTool)):
            return req.default

        loop = asyncio.get_event_loop()
        future: "asyncio.Future[T]" = loop.create_future()

        def respond(value: object) -> None:
            self._pending = None
            if self._listener is not None:
                self._listener()
            if not future.done():
                future.set_result(value)  # type: ignore[arg-type]

        self._pending = PendingRequest(req=req, respond=respond)
        if self._listener is not None:
            self._listener()
        return await future
