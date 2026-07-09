"""Task tracking — the stateful successor to the single `todo_write` tool.

Instead of rewriting the whole list on every call, the model mutates a persisted list
one item at a time: `task_create` adds an item (and returns its id), `task_update`
patches one item by id, `task_list`/`task_get` read it back. This mirrors the Claude
Agent SDK's TaskCreate/TaskUpdate/TaskList/TaskGet split.

The four tools share one `TaskStore`, closed over by `task_tools()`. State lives for
the lifetime of that bundle (i.e. one agent run) — build a fresh bundle per run.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Optional

import pydantic

from curry_leaves.core.tools import Risk, Tool, ToolResult
from curry_leaves.providers.base import Context

Status = Literal["pending", "in_progress", "completed"]

_MARK: dict[str, str] = {"pending": "☐", "in_progress": "▶", "completed": "☑"}


class Task(pydantic.BaseModel):
    id: str
    subject: str
    # Present-tense label shown while in_progress (e.g. "Adding the search tool").
    active_form: Optional[str] = None
    status: Status = "pending"


class TaskStore:
    def __init__(self) -> None:
        self._seq = 0
        self.tasks: list[Task] = []

    def create(self, subject: str, active_form: Optional[str] = None) -> Task:
        self._seq += 1
        task = Task(id=str(self._seq), subject=subject, active_form=active_form, status="pending")
        self.tasks.append(task)
        return task

    def get(self, id: str) -> Optional[Task]:
        for t in self.tasks:
            if t.id == id:
                return t
        return None

    def remove(self, id: str) -> bool:
        for i, t in enumerate(self.tasks):
            if t.id == id:
                del self.tasks[i]
                return True
        return False

    def render(self) -> str:
        """A stable, model-readable snapshot of the whole list."""
        if len(self.tasks) == 0:
            return "(no tasks)"
        done = sum(1 for t in self.tasks if t.status == "completed")
        lines = []
        for t in self.tasks:
            label = t.active_form if t.status == "in_progress" and t.active_form else t.subject
            lines.append(f"{_MARK[t.status]} #{t.id} {label}")
        body = "\n".join(lines)
        return f"{body}\n({done}/{len(self.tasks)} done)"


# ── task_create ──────────────────────────────────────────────────────────────


class CreateArgs(pydantic.BaseModel):
    subject: str = pydantic.Field(
        description="The task, imperative form (e.g. 'Add the search tool')."
    )
    active_form: Optional[str] = pydantic.Field(
        default=None,
        description="Present-tense label shown while active (e.g. 'Adding the search tool').",
    )


class TaskCreateTool:
    name = "task_create"
    risk: Optional[Risk] = "read"
    description = (
        "Add ONE task to your task list and get back its id. Call once per step when planning "
        "multi-step work. Returns the new task's id plus the full current list."
    )
    schema: type[pydantic.BaseModel] = CreateArgs
    timeout: Optional[float] = None

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    async def run(self, args: CreateArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        task = self._store.create(args.subject, args.active_form)
        return ToolResult(content=f"Created task #{task.id}.\n{self._store.render()}")

    async def close(self) -> None:
        pass


def task_create(store: TaskStore) -> Tool[Any]:
    return TaskCreateTool(store)


# ── task_update ──────────────────────────────────────────────────────────────

UpdateStatus = Literal["pending", "in_progress", "completed", "deleted"]


class UpdateArgs(pydantic.BaseModel):
    task_id: str = pydantic.Field(
        description="The id of the task to update (from task_create/task_list)."
    )
    status: Optional[UpdateStatus] = pydantic.Field(
        default=None, description="New status. 'deleted' removes the task from the list."
    )
    subject: Optional[str] = pydantic.Field(default=None, description="Replace the task's subject.")
    active_form: Optional[str] = pydantic.Field(
        default=None, description="Replace the present-tense active label."
    )


class TaskUpdateTool:
    name = "task_update"
    risk: Optional[Risk] = "read"
    description = (
        "Patch ONE task by id: flip its status (pending/in_progress/completed, or 'deleted' to "
        "remove it) and/or edit its text. Keep exactly one task 'in_progress' at a time. Returns "
        "the full current list."
    )
    schema: type[pydantic.BaseModel] = UpdateArgs
    timeout: Optional[float] = None

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    async def run(self, args: UpdateArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        if args.status == "deleted":
            if not self._store.remove(args.task_id):
                return ToolResult(content=f"No task with id #{args.task_id}.", is_error=True)
            return ToolResult(content=f"Deleted task #{args.task_id}.\n{self._store.render()}")

        task = self._store.get(args.task_id)
        if task is None:
            return ToolResult(content=f"No task with id #{args.task_id}.", is_error=True)

        if args.status is not None:
            task.status = args.status
        if args.subject is not None:
            task.subject = args.subject
        if args.active_form is not None:
            task.active_form = args.active_form

        in_progress = sum(1 for t in self._store.tasks if t.status == "in_progress")
        warn = (
            f"\n(warning: {in_progress} tasks in_progress — keep it to one)"
            if in_progress > 1
            else ""
        )
        return ToolResult(content=f"Updated task #{task.id}.\n{self._store.render()}{warn}")

    async def close(self) -> None:
        pass


def task_update(store: TaskStore) -> Tool[Any]:
    return TaskUpdateTool(store)


# ── task_list ────────────────────────────────────────────────────────────────


class ListArgs(pydantic.BaseModel):
    pass


class TaskListTool:
    name = "task_list"
    risk: Optional[Risk] = "read"
    description = "Read back the full current task list with ids and statuses."
    schema: type[pydantic.BaseModel] = ListArgs
    timeout: Optional[float] = None

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    async def run(self, args: ListArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        return ToolResult(content=self._store.render())

    async def close(self) -> None:
        pass


def task_list(store: TaskStore) -> Tool[Any]:
    return TaskListTool(store)


# ── task_get ─────────────────────────────────────────────────────────────────


class GetArgs(pydantic.BaseModel):
    task_id: str = pydantic.Field(description="The id of the task to read.")


class TaskGetTool:
    name = "task_get"
    risk: Optional[Risk] = "read"
    description = "Read one task by id."
    schema: type[pydantic.BaseModel] = GetArgs
    timeout: Optional[float] = None

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    async def run(self, args: GetArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        task = self._store.get(args.task_id)
        if task is None:
            return ToolResult(content=f"No task with id #{args.task_id}.", is_error=True)
        label = task.active_form if task.status == "in_progress" and task.active_form else task.subject
        return ToolResult(content=f"{_MARK[task.status]} #{task.id} {label} ({task.status})")

    async def close(self) -> None:
        pass


def task_get(store: TaskStore) -> Tool[Any]:
    return TaskGetTool(store)


def task_tools() -> list[Tool[Any]]:
    """The task-tracking toolset: `task_create`, `task_update`, `task_list`, `task_get`
    sharing one in-memory store. Build a fresh bundle per agent run so state doesn't
    leak across runs.
    """
    store = TaskStore()
    return [task_create(store), task_update(store), task_list(store), task_get(store)]
