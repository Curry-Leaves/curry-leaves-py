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
import json
from pathlib import Path
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


def _label(task: "Task") -> str:
    """The active-form label while in_progress (if set), else the subject."""
    if task.status == "in_progress" and task.active_form:
        return task.active_form
    return task.subject


class TaskStore:
    """In-memory by default. Pass `path` to persist across runs (e.g. one file per chat
    session) — every mutation rewrites it, and it's loaded back in on construction.

    `reset_if_done`: a fully-completed list on load is treated as belonging to the PRIOR
    request, not the new one — it's cleared so a fresh plan starts empty rather than
    carrying a finished checklist into unrelated work.
    """

    def __init__(self, path: Optional[Path] = None, reset_if_done: bool = True) -> None:
        self._seq = 0
        self.tasks: list[Task] = []
        self._path = path
        if path is not None and path.exists():
            data = json.loads(path.read_text())
            self.tasks = [Task(**t) for t in data.get("tasks", [])]
            self._seq = data.get("seq", len(self.tasks))
            if reset_if_done and self.tasks and all(t.status == "completed" for t in self.tasks):
                self.tasks = []
                self._save()

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"seq": self._seq, "tasks": [t.model_dump() for t in self.tasks]}))
        tmp.replace(self._path)

    def create(self, subject: str, active_form: Optional[str] = None) -> Task:
        self._seq += 1
        task = Task(id=str(self._seq), subject=subject, active_form=active_form, status="pending")
        self.tasks.append(task)
        self._save()
        return task

    def get(self, id: str) -> Optional[Task]:
        for t in self.tasks:
            if t.id == id:
                return t
        return None

    def update(self, id: str, **patch: Any) -> Optional[Task]:
        task = self.get(id)
        if task is None:
            return None
        for k, v in patch.items():
            if v is not None:
                setattr(task, k, v)
        self._save()
        return task

    def remove(self, id: str) -> bool:
        for i, t in enumerate(self.tasks):
            if t.id == id:
                del self.tasks[i]
                self._save()
                return True
        return False

    def render(self) -> str:
        """A stable, model-readable snapshot of the whole list."""
        if len(self.tasks) == 0:
            return "(no tasks)"
        done = sum(1 for t in self.tasks if t.status == "completed")
        lines = []
        for t in self.tasks:
            lines.append(f"{_MARK[t.status]} #{t.id} {_label(t)}")
        body = "\n".join(lines)
        return f"{body}\n({done}/{len(self.tasks)} done)"


class _TaskTool:
    """Shared base for the four task tools: they all close over one `TaskStore` and have
    no resources to tear down, so the `store`-holding constructor lives here once."""

    def __init__(self, store: TaskStore) -> None:
        self._store = store


# ── task_create ──────────────────────────────────────────────────────────────


class TaskSpec(pydantic.BaseModel):
    subject: str = pydantic.Field(
        description="The task, imperative form (e.g. 'Add the search tool')."
    )
    active_form: Optional[str] = pydantic.Field(
        default=None,
        description="Present-tense label shown while active (e.g. 'Adding the search tool').",
    )


class CreateArgs(pydantic.BaseModel):
    tasks: list[TaskSpec] = pydantic.Field(
        min_length=1,
        description=(
            "The tasks to add, in order. Pass the WHOLE plan in one call — one entry per "
            "step. A single-item list is fine for one-off additions."
        ),
    )


class TaskCreateTool(_TaskTool):
    name = "task_create"
    risk: Optional[Risk] = "read"
    description = (
        "Add one or more tasks to your task list in a single call. When planning multi-step "
        "work, pass the entire plan as the `tasks` array at once rather than calling this "
        "repeatedly. Returns the new task ids plus the full current list."
    )
    schema: type[pydantic.BaseModel] = CreateArgs
    timeout: Optional[float] = None

    async def run(self, args: CreateArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        created = [self._store.create(spec.subject, spec.active_form) for spec in args.tasks]
        ids = ", ".join(f"#{t.id}" for t in created)
        label = "task" if len(created) == 1 else "tasks"
        return ToolResult(content=f"Created {label} {ids}.\n{self._store.render()}")


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


class TaskUpdateTool(_TaskTool):
    name = "task_update"
    risk: Optional[Risk] = "read"
    description = (
        "Patch ONE task by id: flip its status (pending/in_progress/completed, or 'deleted' to "
        "remove it) and/or edit its text. Keep exactly one task 'in_progress' at a time. Returns "
        "the full current list."
    )
    schema: type[pydantic.BaseModel] = UpdateArgs
    timeout: Optional[float] = None

    async def run(self, args: UpdateArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        if args.status == "deleted":
            if not self._store.remove(args.task_id):
                return ToolResult(content=f"No task with id #{args.task_id}.", is_error=True)
            return ToolResult(content=f"Deleted task #{args.task_id}.\n{self._store.render()}")

        task = self._store.update(
            args.task_id,
            status=args.status,
            subject=args.subject,
            active_form=args.active_form,
        )
        if task is None:
            return ToolResult(content=f"No task with id #{args.task_id}.", is_error=True)

        in_progress = sum(1 for t in self._store.tasks if t.status == "in_progress")
        warn = (
            f"\n(warning: {in_progress} tasks in_progress — keep it to one)"
            if in_progress > 1
            else ""
        )
        return ToolResult(content=f"Updated task #{task.id}.\n{self._store.render()}{warn}")


def task_update(store: TaskStore) -> Tool[Any]:
    return TaskUpdateTool(store)


# ── task_list ────────────────────────────────────────────────────────────────


class ListArgs(pydantic.BaseModel):
    pass


class TaskListTool(_TaskTool):
    name = "task_list"
    risk: Optional[Risk] = "read"
    description = "Read back the full current task list with ids and statuses."
    schema: type[pydantic.BaseModel] = ListArgs
    timeout: Optional[float] = None

    async def run(self, args: ListArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        return ToolResult(content=self._store.render())


def task_list(store: TaskStore) -> Tool[Any]:
    return TaskListTool(store)


# ── task_get ─────────────────────────────────────────────────────────────────


class GetArgs(pydantic.BaseModel):
    task_id: str = pydantic.Field(description="The id of the task to read.")


class TaskGetTool(_TaskTool):
    name = "task_get"
    risk: Optional[Risk] = "read"
    description = "Read one task by id."
    schema: type[pydantic.BaseModel] = GetArgs
    timeout: Optional[float] = None

    async def run(self, args: GetArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        task = self._store.get(args.task_id)
        if task is None:
            return ToolResult(content=f"No task with id #{args.task_id}.", is_error=True)
        return ToolResult(content=f"{_MARK[task.status]} #{task.id} {_label(task)} ({task.status})")


def task_get(store: TaskStore) -> Tool[Any]:
    return TaskGetTool(store)


def task_tools(store: Optional[TaskStore] = None) -> list[Tool[Any]]:
    """The task-tracking toolset: `task_create`, `task_update`, `task_list`, `task_get`
    sharing one store. Pass a `TaskStore` (e.g. one backed by a per-session file) to
    reuse/persist state; omit it for a fresh in-memory store scoped to this call.
    """
    store = store or TaskStore()
    return [task_create(store), task_update(store), task_list(store), task_get(store)]
