"""curry-leaves — a small, extensible, general-purpose agent kernel.

The public API is intentionally tiny — enough to build an agent::

    from curry_leaves import Agent, Runner
    agent = Agent(model="claude-sonnet-4-5", instructions="...")
    result = await Runner(agent).run("Summarize README.md in three bullets.")
    print(result.output_text)
"""

VERSION = "1.4.0"

# ── core: the definition, the driver, the engine ─────────────────────────────
from .core.agent import Agent, AgentOptions, AgentTool
from .runner import MAX_AGENT_DEPTH, RunConfig, Runner, RunResult
from .core.loop import Interrupt, LoopConfig, agent_loop, stream_assistant

# ── the tool system ───────────────────────────────────────────────────────────
from .core.tools import (
    MAX_RESULT_CHARS,
    Risk,
    Tool,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    cap_result_text,
    make_executor,
)

# ── the neutral message model + events ───────────────────────────────────────
from .core.messages import (
    AudioBlock,
    Content,
    Cost,
    FileBlock,
    ImageBlock,
    Message,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
    AssistantMessage,
    add_cost,
    add_usage,
    assistant_text,
    empty_assistant,
    empty_cost,
    empty_usage,
    text_of,
    tool_result_text,
    user_audio,
    user_file,
    user_image,
    user_text,
)
from .core.events import (
    AgentEnd,
    AgentEvent,
    ApprovalEvent,
    CompactionEvent,
    Delta,
    ElisionEvent,
    ErrorEvent,
    HandoffEvent,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    SubagentActivity,
    ThinkingEvent,
    ToolEnd,
    ToolStart,
    TurnEnd,
    TurnStart,
    ev,
    flatten,
)

# ── host + blobs ──────────────────────────────────────────────────────────────
from .core.host import ApproveTool, ApprovalChoice, AskUser, DefaultHost, Host, Request, SubagentHost
from .permission import (
    AuthorizeContext,
    AutoApprove,
    PermissionEngine,
    PermissionOptions,
    Verdict,
    contained_approval,
)
from .settings import (
    DEFAULT_LOCAL_MODEL,
    Settings,
    add_global_approval,
    auto_hosts,
    global_approvals,
    load_settings,
    resolve_default_model,
    save_model_choice,
    user_settings_path,
)
from .core.blobs import BlobStore

# ── providers + catalog ───────────────────────────────────────────────────────
from .providers.base import (
    Context,
    Model,
    ModelSettings,
    Provider,
    StreamChunk,
    StreamDone,
    StreamEvent,
    StreamOpts,
    ToolSchema,
    make_model,
    settings_to_opts,
)
from .providers.anthropic import AnthropicProvider
from .providers.openai import OllamaProvider, OpenAIProvider, OpenAIProviderOptions
from .providers.factory import infer_provider, provider_for, provider_name_for_model
from .catalog import CATALOG, LoadCatalogOptions, ModelInfo, compute_cost, load_catalog, lookup, resolve_model

# ── prompt, thinking, skills ──────────────────────────────────────────────────
from .prompt import (
    CODING_IDENTITY,
    DEFAULT_IDENTITY,
    BuildPromptOptions,
    ContextFile,
    Environment,
    build_system_prompt,
    resolve_identity,
)
from .thinking import (
    DEFAULT_CLASSIFIER_SYSTEM,
    AutoThinking,
    AutoThinkingOptions,
    Classifier,
    Effort,
    ThinkingConfig,
    thinking_budget,
)
from .skills import Skill, SkillRegistry
from .compaction import (
    SUMMARY_SYSTEM,
    Compactor,
    CompactionConfig,
    CompactionOutcome,
    estimate_tokens,
)
from .elision import Elider, ElisionConfig, ElisionOutcome

# ── retry ─────────────────────────────────────────────────────────────────────
from .util.retry import DefaultRetryPolicy, HttpError, RetryPolicy
from .util.paths import (
    home,
    repo_root,
    session_dir,
    session_meta_file,
    session_transcript_file,
    sessions_dir,
    set_home,
)
from .session import (
    FileSessionStore,
    MemorySessionStore,
    NullSessionStore,
    SessionMeta,
    SessionStore,
    fork_session,
    load_meta,
    load_transcript,
    open_session,
    transcript_to_messages,
    user_turn_offsets,
)

# ── builtin agents + presets ──────────────────────────────────────────────────
from .agents import Plan, explore_agent, plan_agent
from .presets import coding_tools, web_tools

# ── individual tool factories (compose your own toolset) ─────────────────────
from .tools.read import read_tool
from .tools.write import write_tool
from .tools.edit import edit_tool
from .tools.find import find_tool
from .tools.search import search_tool
from .tools.bash import bash_tool
from .tools.tasks import task_tools
from .tools.ask import ask_tool
from .tools.current_time import current_time_tool
from .tools.web import web_fetch_tool, web_search_tool
from .tools.search_tools import SearchToolsTool
from .tools.task import Spawn, TaskTool
from .tools.transfer import Transfer, TransferTool

__all__ = [
    "VERSION",
    # core
    "Agent",
    "AgentOptions",
    "AgentTool",
    "Runner",
    "RunConfig",
    "RunResult",
    "MAX_AGENT_DEPTH",
    "agent_loop",
    "stream_assistant",
    "Interrupt",
    "LoopConfig",
    # tools
    "Tool",
    "ToolResult",
    "Risk",
    "ToolRegistry",
    "make_executor",
    "cap_result_text",
    "MAX_RESULT_CHARS",
    "ToolExecutor",
    # messages
    "TextBlock",
    "ThinkingBlock",
    "ToolCallBlock",
    "ImageBlock",
    "AudioBlock",
    "FileBlock",
    "Content",
    "Cost",
    "empty_cost",
    "add_cost",
    "Usage",
    "empty_usage",
    "add_usage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "Message",
    "StopReason",
    "user_text",
    "user_image",
    "user_audio",
    "user_file",
    "assistant_text",
    "empty_assistant",
    "tool_result_text",
    "text_of",
    # events
    "AgentEvent",
    "Delta",
    "MessageStart",
    "MessageUpdate",
    "MessageEnd",
    "ToolStart",
    "ToolEnd",
    "TurnStart",
    "TurnEnd",
    "ErrorEvent",
    "ThinkingEvent",
    "HandoffEvent",
    "CompactionEvent",
    "ElisionEvent",
    "ApprovalEvent",
    "AgentEnd",
    "SubagentActivity",
    "ev",
    "flatten",
    # host + blobs
    "Host",
    "Request",
    "AskUser",
    "ApproveTool",
    "ApprovalChoice",
    "DefaultHost",
    "SubagentHost",
    "PermissionEngine",
    "contained_approval",
    "Verdict",
    "PermissionOptions",
    "AuthorizeContext",
    "AutoApprove",
    "load_settings",
    "global_approvals",
    "auto_hosts",
    "add_global_approval",
    "user_settings_path",
    "resolve_default_model",
    "save_model_choice",
    "DEFAULT_LOCAL_MODEL",
    "Settings",
    "BlobStore",
    # providers + catalog
    "Provider",
    "Model",
    "ModelSettings",
    "StreamOpts",
    "StreamEvent",
    "StreamChunk",
    "StreamDone",
    "Context",
    "ToolSchema",
    "make_model",
    "settings_to_opts",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "OpenAIProviderOptions",
    "infer_provider",
    "provider_for",
    "provider_name_for_model",
    "CATALOG",
    "ModelInfo",
    "LoadCatalogOptions",
    "lookup",
    "resolve_model",
    "compute_cost",
    "load_catalog",
    # prompt, thinking, skills
    "build_system_prompt",
    "resolve_identity",
    "DEFAULT_IDENTITY",
    "CODING_IDENTITY",
    "ContextFile",
    "Environment",
    "BuildPromptOptions",
    "AutoThinking",
    "AutoThinkingOptions",
    "Classifier",
    "ThinkingConfig",
    "DEFAULT_CLASSIFIER_SYSTEM",
    "Effort",
    "thinking_budget",
    "SkillRegistry",
    "Skill",
    "Compactor",
    "CompactionConfig",
    "CompactionOutcome",
    "estimate_tokens",
    "SUMMARY_SYSTEM",
    "Elider",
    "ElisionConfig",
    "ElisionOutcome",
    # retry, paths, sessions
    "DefaultRetryPolicy",
    "HttpError",
    "RetryPolicy",
    "set_home",
    "home",
    "repo_root",
    "sessions_dir",
    "session_dir",
    "session_meta_file",
    "session_transcript_file",
    "SessionStore",
    "SessionMeta",
    "FileSessionStore",
    "MemorySessionStore",
    "NullSessionStore",
    "open_session",
    "fork_session",
    "load_meta",
    "load_transcript",
    "transcript_to_messages",
    "user_turn_offsets",
    # builtin agents + presets
    "explore_agent",
    "plan_agent",
    "Plan",
    "coding_tools",
    "web_tools",
    # individual tool factories
    "read_tool",
    "write_tool",
    "edit_tool",
    "find_tool",
    "search_tool",
    "bash_tool",
    "task_tools",
    "ask_tool",
    "current_time_tool",
    "web_fetch_tool",
    "web_search_tool",
    "SearchToolsTool",
    "TaskTool",
    "Spawn",
    "TransferTool",
    "Transfer",
]
