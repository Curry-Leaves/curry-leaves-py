"""PermissionEngine — the gate every tool call passes through.

There are NO modes. An agent declares a per-tool verdict map (with a `"*"` catch-all); the
engine combines that with the tool's risk and any standing approvals to decide, per call:

    allow → run   ·   ask → prompt the user (via the host)   ·   deny → block

Resolution order (first match wins):
  1. permissions[tool] == "deny"            → DENY   (absolute — approvals can't override)
  2. permissions[tool] == "allow"           → ALLOW
  3. tool in approvals (global ∪ session)   → ALLOW  (a prior "always")
  4. permissions[tool] == "ask"             → ASK
  5. permissions["*"]                       → its verdict (fallback default)
  6. risk fallback                          → read: ALLOW · else: ASK

ONE engine is shared across a run (root + subagents) so its approval store is unified; the
per-agent `permissions` map and the `host` are passed PER CALL (via AuthorizeContext) so a
subagent never clobbers the parent's. Approvals are two-tier: `global` (persisted to
settings.json by the frontend via `on_global_approve`) and `session` (in-memory this run,
stamped into session meta for audit). No host — or a host that returns the default `deny` —
fails closed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal
from urllib.parse import urlparse

from curry_leaves.core.events import ev
from curry_leaves.core.host import ApproveTool, Host
from curry_leaves.core.tools import AuthorizeResult, Risk

Verdict = Literal["allow", "ask", "deny"]
_Decision = Literal["allow", "ask", "deny"]

# Predicate: is a would-be-prompted call CONTAINED enough to auto-approve? (see
# contained_approval)
AutoApprove = Callable[[str, Risk, "dict[str, object]"], bool]


@dataclass
class AuthorizeContext:
    """Per-call context the Runner supplies: the ACTIVE agent's verdicts + the host to
    prompt on.
    """

    permissions: dict[str, Verdict]
    host: Host | None


@dataclass
class PermissionOptions:
    # Standing approvals loaded from settings.json (global, all sessions).
    global_approvals: Iterable[str] = field(default_factory=list)
    # Persist a global "always" grant (the frontend writes settings.json).
    on_global_approve: Callable[[str], None] | None = None
    # Auto-approve a call the policy would otherwise PROMPT for, when its effect is
    # contained (e.g. a write inside the repo, a fetch to a known host). Returns true →
    # grant without asking. The frontend supplies this (it knows the repo root + host
    # list) and can gate it on/off. See `contained_approval`.
    auto_approve: AutoApprove | None = None


class PermissionEngine:
    def __init__(self, opts: PermissionOptions | None = None) -> None:
        opts = opts if opts is not None else PermissionOptions()
        self._global_approvals: set[str] = set(opts.global_approvals or [])
        self._session_approvals: set[str] = set()
        self._on_global_approve = opts.on_global_approve
        self._auto_approve = opts.auto_approve

    @property
    def session_grants(self) -> list[str]:
        """Session-scoped grants made this run (stamped into session meta for audit)."""
        return list(self._session_approvals)

    def _decide(self, tool: str, risk: Risk, permissions: dict[str, Verdict]) -> _Decision:
        v = permissions.get(tool)
        if v == "deny":
            return "deny"  # absolute
        if v == "allow":
            return "allow"
        if tool in self._global_approvals or tool in self._session_approvals:
            return "allow"
        if v == "ask":
            return "ask"
        star = permissions.get("*")
        if star:
            return star
        return "allow" if risk == "read" else "ask"  # risk fallback

    async def authorize(
        self,
        tool: str,
        risk: Risk,
        args: dict[str, object],
        ctx: AuthorizeContext,
    ) -> AuthorizeResult:
        """Resolve a call to (allowed, reason). Prompts via the host when the verdict is
        `ask`.
        """
        decision = self._decide(tool, risk, ctx.permissions)
        if decision == "allow":
            return AuthorizeResult(ok=True, reason="")
        if decision == "deny":
            if ctx.host is not None:
                ctx.host.emit(ev.approval(tool, risk, False, "deny"))
            return AuthorizeResult(ok=False, reason="blocked by permission policy")
        # ASK — but auto-approve first if the call is contained (auto mode). Not
        # remembered (per-call).
        if self._auto_approve is not None and self._auto_approve(tool, risk, args):
            if ctx.host is not None:
                ctx.host.emit(ev.approval(tool, risk, True, "auto"))
            return AuthorizeResult(ok=True, reason="")
        # Otherwise route to the host for a choice.
        req = ApproveTool(tool=tool, args=args, risk=risk, reason="", default="deny")
        choice = await ctx.host.request(req) if ctx.host is not None else "deny"
        if choice == "always":
            self._global_approvals.add(tool)
            if self._on_global_approve is not None:
                self._on_global_approve(tool)
        elif choice == "session":
            self._session_approvals.add(tool)
        granted = choice != "deny"
        if ctx.host is not None:
            ctx.host.emit(ev.approval(tool, risk, granted, choice))
        return AuthorizeResult(ok=granted, reason="" if granted else "denied by user")


# ── contained-auto-approval ──────────────────────────────────────────────────


def _path_args(args: dict[str, object]) -> list[str]:
    """Gather filesystem-path args by common key names (path/file/filepath + array
    `paths`).
    """
    out: list[str] = []
    for key in ("path", "file", "filepath", "filePath"):
        v = args.get(key)
        if isinstance(v, str):
            out.append(v)
    paths = args.get("paths")
    if isinstance(paths, list):
        out.extend(p for p in paths if isinstance(p, str))
    return out


def _is_inside(root: str, p: str) -> bool:
    """Is `p` inside `root` (or equal to it)? Resolves relative paths against root.

    Compares realpaths, not abspaths: a symlink inside the repo pointing outside it must
    not count as contained, or auto-approved writes could silently escape the sandbox.
    """
    abs_p = os.path.abspath(p) if os.path.isabs(p) else os.path.abspath(os.path.join(root, p))
    rel = os.path.relpath(os.path.realpath(abs_p), os.path.realpath(root))
    return rel == "." or (not rel.startswith("..") and not os.path.isabs(rel))


def contained_approval(root: str, hosts: Iterable[str] | None = None) -> AutoApprove:
    """Build an AutoApprove predicate for "auto" mode — grant a would-be-prompted call
    ONLY when its effect is contained:
      - write  → every path arg resolves inside `root` (the repo)
      - network→ the URL's host is in `hosts`
      - exec   → never (a shell command's effects can't be verified statically)
      - read   → yes (harmless; usually already allowed anyway)
    Anything else (unknown shape, path/host escaping the sandbox) → false → still
    prompts.
    """
    resolved_root = os.path.abspath(root)
    host_set = set(hosts) if hosts is not None else set()

    def predicate(_tool: str, risk: Risk, args: dict[str, object]) -> bool:
        if risk == "read":
            return True
        if risk == "exec":
            return False
        if risk == "network":
            url = args.get("url")
            if not isinstance(url, str):
                return False
            try:
                hostname = urlparse(url).hostname
            except ValueError:
                return False
            return hostname is not None and hostname in host_set
        if risk == "write":
            paths = _path_args(args)
            return len(paths) > 0 and all(_is_inside(resolved_root, p) for p in paths)
        return False

    return predicate
