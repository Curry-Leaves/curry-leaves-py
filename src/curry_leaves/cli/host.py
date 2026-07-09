"""CliHost — the plain REPL's interactive host adapter.

Extends DefaultHost (keeps its event emit/subscribe) and implements `request` by prompting on
the terminal — one adapter serving BOTH request kinds: `ask_user` (a question) and
`approve_tool` (a permission prompt). It reuses the REPL's single `Prompter` (see chat.py), so
only one question is ever active at a time (the main loop is awaiting a turn when the host
prompts).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypeVar

from curry_leaves.core.host import ApprovalChoice, ApproveTool, AskUser, DefaultHost, Request

from .theme import CYAN, DIM, RESET, YELLOW

if TYPE_CHECKING:
    from .chat import Prompter

T = TypeVar("T")


class CliHost(DefaultHost):
    def __init__(self, prompter: "Prompter") -> None:
        super().__init__()
        self._prompter = prompter

    async def request(self, req: Request[T]) -> T:
        if isinstance(req, AskUser):
            return await self._ask(req)  # type: ignore[return-value]
        if isinstance(req, ApproveTool):
            return await self._approve(req)  # type: ignore[return-value]
        return req.default

    async def _ask(self, req: AskUser) -> str:
        opts = f" {DIM}[{' / '.join(req.options)}]{RESET}" if req.options else ""
        raw = await self._prompter.question(f"\n{CYAN}❓ {req.question}{RESET}{opts}\n{CYAN}› {RESET}")
        ans = raw.strip()
        return ans or req.default

    async def _approve(self, req: ApproveTool) -> ApprovalChoice:
        args = json.dumps(req.args or {})
        self._prompter.write(f"\n{YELLOW}⚠ allow {req.tool}{RESET} {DIM}({req.risk}) {args[:100]}{RESET}\n")
        raw = await self._prompter.question(f"{DIM}  [y = once · s = this session · a = always · n = no] › {RESET}")
        ans = raw.strip().lower()
        # Engine persists an "always" grant via its on_global_approve hook; the host just reports the choice.
        if ans in ("a", "always"):
            return "always"
        if ans in ("s", "session"):
            return "session"
        if ans in ("y", "yes", "once"):
            return "once"
        return "deny"
