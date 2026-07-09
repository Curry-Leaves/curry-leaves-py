"""The `bash` tool: run a shell command and return its combined stdout+stderr and exit code."""

from __future__ import annotations

import asyncio
import os
import signal as _signal
from typing import Any

import pydantic

from curry_leaves.core.tools import Risk, Tool, ToolResult
from curry_leaves.providers.base import Context

MAX_OUTPUT_CHARS = 30_000
MAX_TIMEOUT_SECONDS = 600


_POSIX = os.name == "posix"


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the command's whole process group — killing only the `sh -c` shell would
    leave its children (test runners, servers, pipelines) running and holding the pipe."""
    if _POSIX:
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass  # group already gone or not ours → fall back to the shell itself
    if proc.returncode is None:
        proc.kill()


class BashArgs(pydantic.BaseModel):
    command: str = pydantic.Field(description="The shell command to run.")
    timeout_seconds: int = pydantic.Field(
        default=60,
        description="Kill the command after this many SECONDS (not ms; capped at 600).",
    )


class BashTool:
    """Structurally satisfies the `Tool` protocol (see core/tools.py)."""

    name = "bash"
    description = (
        "Run a shell command in the working directory and return its combined stdout+stderr and "
        "exit code. Use for builds, tests, git, file listings, etc."
    )
    schema: type[pydantic.BaseModel] = BashArgs
    risk: Risk | None = "exec"
    timeout: float | None = None

    async def run(self, args: BashArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        timeout = min(max(1, args.timeout_seconds), MAX_TIMEOUT_SECONDS)

        proc = await asyncio.create_subprocess_shell(
            args.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=_POSIX,  # own process group, so a kill reaches grandchildren too
        )

        # Tie the outer abort/steering signal to killing the child, same as the TS
        # AbortSignal listener — mirrored exactly so a steering interrupt cancels an
        # in-flight command instead of leaving it running detached.
        async def watch_outer() -> None:
            await signal.wait()
            if proc.returncode is None:
                _kill_tree(proc)

        watcher = asyncio.ensure_future(watch_outer())

        timed_out = False
        try:
            assert proc.stdout is not None
            try:
                out_bytes = await asyncio.wait_for(proc.stdout.read(), timeout=timeout)
                await proc.wait()
            except asyncio.TimeoutError:
                timed_out = True
                _kill_tree(proc)
                await proc.wait()
                out_bytes = b""
        except OSError as e:
            return ToolResult(content=f"Failed to run command: {e}", is_error=True)
        finally:
            watcher.cancel()
            try:
                await watcher
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        if timed_out:
            return ToolResult(
                content=f"Command timed out after {timeout}s: {args.command}", is_error=True
            )

        output = out_bytes.decode("utf-8", errors="replace")
        if len(output) > MAX_OUTPUT_CHARS:
            preview = output[:MAX_OUTPUT_CHARS]
            if ctx.blobs is not None:
                bid = ctx.blobs.put_text(output)
                output = (
                    f"{preview}\n... [truncated — full {len(output)} chars stored at "
                    f"artifact://{bid}; read it with offset/limit]"
                )
            else:
                output = f"{preview}\n... [output truncated]"

        rc = proc.returncode if proc.returncode is not None else 0
        return ToolResult(content=f"(exit {rc})\n{output or '(no output)'}", is_error=rc != 0)

    async def close(self) -> None:
        pass


def bash_tool() -> Tool[Any]:
    return BashTool()
