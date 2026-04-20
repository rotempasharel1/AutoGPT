"""Claude Agent SDK service layer for CoPilot chat completions."""

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple, cast

if TYPE_CHECKING:
    from backend.copilot.permissions import CopilotPermissions

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from langfuse import propagate_attributes
from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk
from pydantic import BaseModel

from backend.copilot.context import get_workspace_manager
from backend.copilot.permissions import apply_tool_permissions
from backend.data.redis_client import get_redis_async
from backend.executor.cluster_lock import AsyncClusterLock
from backend.util.exceptions import NotFoundError
from backend.util.settings import Settings

from ..config import ChatConfig
from ..constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    COPILOT_SYSTEM_PREFIX,
    FRIENDLY_TRANSIENT_MSG,
    is_transient_api_error,
)
from ..context import encode_cwd_for_cli
from ..model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    update_session_title,
    upsert_chat_session,
)
from ..prompting import get_sdk_supplement
from ..response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamHeartbeat,
    StreamStart,
    StreamStatus,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
    StreamUsage,
)
from ..service import (
    _build_system_prompt,
    _generate_session_title,
    _is_langfuse_configured,
)
from ..token_tracking import persist_and_record_usage
from ..tools.e2b_sandbox import get_or_create_sandbox, pause_sandbox_direct
from ..tools.sandbox import WORKSPACE_PREFIX, make_session_path
from ..tracking import track_user_message
from .compaction import CompactionTracker, filter_compaction_messages
from .response_adapter import SDKResponseAdapter
from .security_hooks import create_security_hooks
from .subscription import validate_subscription as _validate_claude_code_subscription
from .tool_adapter import (
    cancel_pending_tool_tasks,
    create_copilot_mcp_server,
    get_copilot_tool_names,
    get_sdk_disallowed_tools,
    pre_launch_tool_call,
    reset_stash_event,
    reset_tool_failure_counters,
    set_execution_context,
    wait_for_stash,
)
from .transcript import (
    _run_compression,
    cleanup_stale_project_dirs,
    compact_transcript,
    download_transcript,
    read_compacted_entries,
    upload_transcript,
    validate_transcript,
    write_transcript_to_tempfile,
)
from .transcript_builder import TranscriptBuilder

logger = logging.getLogger(__name__)
config = ChatConfig()


# On context-size errors the SDK query is retried with progressively
# less context: (1) original transcript → (2) compacted transcript →
# (3) no transcript (DB messages only).
# Non-context errors (network, auth, rate-limit) are NOT retried.
_MAX_STREAM_ATTEMPTS = 3

# Hard circuit breaker: abort the stream if the model sends this many
# consecutive tool calls with empty parameters (a sign of context
# saturation or serialization failure). Empty input ({}) is never
# legitimate — even one is suspicious, three is conclusive.
_EMPTY_TOOL_CALL_LIMIT = 3

# User-facing error shown when the empty-tool-call circuit breaker trips.
_CIRCUIT_BREAKER_ERROR_MSG = (
    "AutoPilot was unable to complete the tool call "
    "— this usually happens when the response is "
    "too large to fit in a single tool call. "
    "Try breaking your request into smaller parts."
)

# Patterns that indicate the prompt/request exceeds the model's context limit.
# Matched case-insensitively against the full exception chain.
_PROMPT_TOO_LONG_PATTERNS: tuple[str, ...] = (
    "prompt is too long",
    "request too large",
    "maximum context length",
    "context_length_exceeded",
    "input tokens exceed",
    "input is too long",
    "content length exceeds",
)


# Map raw SDK error patterns to user-friendly messages.
# Matched case-insensitively; first match wins.
_FRIENDLY_ERROR_MAP: tuple[tuple[str, str], ...] = (
    ("authentication", "Authentication failed. Please check your API key."),
    ("invalid api key", "Authentication failed. Please check your API key."),
    ("unauthorized", "Authentication failed. Please check your API key."),
    ("rate limit", "Rate limit exceeded. Please wait a moment and try again."),
    ("overloaded", "The AI service is currently overloaded. Please try again shortly."),
    ("server error", "The AI service encountered an internal error. Please retry."),
    ("timeout", "The request timed out. Please try again."),
    ("connection", "Connection error. Please check your network and retry."),
)


def _friendly_error_text(raw: str) -> str:
    """Map a raw SDK error string to a user-friendly message.

    Returns the mapped message if a known pattern is found, otherwise
    returns a generic sanitized version of the raw error.
    """
    lower = raw.lower()
    for pattern, friendly in _FRIENDLY_ERROR_MAP:
        if pattern in lower:
            return friendly
    return f"SDK stream error: {raw}"


def _is_prompt_too_long(err: BaseException) -> bool:
    """Return True if *err* indicates the prompt exceeds the model's limit."""
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        msg = str(current).lower()
        if any(p in msg for p in _PROMPT_TOO_LONG_PATTERNS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_tool_only_message(sdk_msg: object) -> bool:
    """Return True if *sdk_msg* is an AssistantMessage containing only ToolUseBlocks."""
    return (
        isinstance(sdk_msg, AssistantMessage)
        and bool(sdk_msg.content)
        and all(isinstance(b, ToolUseBlock) for b in sdk_msg.content)
    )


class ReducedContext(NamedTuple):
    builder: TranscriptBuilder
    use_resume: bool
    resume_file: str | None
    transcript_lost: bool
    tried_compaction: bool


@dataclass
class _TokenUsage:
    """Token usage accumulators for a single turn."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
        self.cost_usd = None


@dataclass
class _RetryState:
    """Mutable state passed to `_run_stream_attempt` instead of closures."""

    options: ClaudeAgentOptions
    query_message: str
    was_compacted: bool
    use_resume: bool
    resume_file: str | None
    transcript_msg_count: int
    adapter: SDKResponseAdapter
    transcript_builder: TranscriptBuilder
    usage: _TokenUsage


@dataclass
class _StreamContext:
    """Per-request variables shared across all retry attempts."""

    session: ChatSession
    session_id: str
    log_prefix: str
    sdk_cwd: str
    current_message: str
    file_ids: list[str] | None
    message_id: str
    attachments: "PreparedAttachments"
    compaction: CompactionTracker
    lock: AsyncClusterLock


async def _reduce_context(
    transcript_content: str,
    tried_compaction: bool,
    session_id: str,
    sdk_cwd: str,
    log_prefix: str,
) -> ReducedContext:
    """Prepare reduced context for a retry attempt."""
    if transcript_content and not tried_compaction:
        compacted = await compact_transcript(
            transcript_content, model=config.model, log_prefix=log_prefix
        )
        if (
            compacted
            and compacted != transcript_content
            and validate_transcript(compacted)
        ):
            logger.info("%s Using compacted transcript for retry", log_prefix)
            tb = TranscriptBuilder()
            tb.load_previous(compacted, log_prefix=log_prefix)
            resume_file = await asyncio.to_thread(
                write_transcript_to_tempfile, compacted, session_id, sdk_cwd
            )
            if resume_file:
                return ReducedContext(tb, True, resume_file, False, True)
            logger.warning("%s Failed to write compacted transcript", log_prefix)
        logger.warning("%s Compaction failed, dropping transcript", log_prefix)

    logger.warning("%s Dropping transcript, rebuilding from DB messages", log_prefix)
    return ReducedContext(TranscriptBuilder(), False, None, True, True)


def _append_error_marker(
    session: ChatSession | None,
    display_msg: str,
    *,
    retryable: bool = False,
) -> None:
    """Append a copilot error marker to *session* so it persists across refresh."""
    if session is None:
        return
    prefix = COPILOT_RETRYABLE_ERROR_PREFIX if retryable else COPILOT_ERROR_PREFIX
    session.messages.append(
        ChatMessage(role="assistant", content=f"{prefix} {display_msg}")
    )


def _setup_langfuse_otel() -> None:
    """Configure OTEL tracing for the Claude Agent SDK → Langfuse."""
    if not _is_langfuse_configured():
        return

    try:
        settings = Settings()
        pk = settings.secrets.langfuse_public_key
        sk = settings.secrets.langfuse_secret_key
        host = settings.secrets.langfuse_host

        creds = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
        os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", f"{host}/api/public/otel")
        os.environ.setdefault(
            "OTEL_EXPORTER_OTLP_HEADERS", f"Authorization=Basic {creds}"
        )

        tracing_env = settings.secrets.langfuse_tracing_environment
        os.environ.setdefault(
            "OTEL_RESOURCE_ATTRIBUTES",
            f"langfuse.environment={tracing_env}",
        )

        configure_claude_agent_sdk(tags=["sdk"])
        logger.info(
            "OTEL tracing configured for Claude Agent SDK → %s [%s]", host, tracing_env
        )
    except Exception:
        logger.warning("OTEL setup skipped — failed to configure", exc_info=True)


_setup_langfuse_otel()

_background_tasks: set[asyncio.Task[Any]] = set()

_SDK_CWD_PREFIX = WORKSPACE_PREFIX

_last_sweep_time: float = 0.0
_SWEEP_INTERVAL_SECONDS = 300

_HEARTBEAT_INTERVAL = 10.0

STREAM_LOCK_PREFIX = "copilot:stream:lock:"


async def _iter_sdk_messages(
    client: ClaudeSDKClient,
) -> AsyncGenerator[Any, None]:
    """Yield SDK messages with heartbeat-based timeouts."""
    msg_iter = client.receive_response().__aiter__()
    pending_task: asyncio.Task[Any] | None = None

    async def _next_msg() -> Any:
        return await msg_iter.__anext__()

    try:
        while True:
            if pending_task is None:
                pending_task = asyncio.create_task(_next_msg())

            done, _ = await asyncio.wait({pending_task}, timeout=_HEARTBEAT_INTERVAL)

            if not done:
                yield None
                continue

            pending_task = None
            try:
                yield done.pop().result()
            except StopAsyncIteration:
                return
    finally:
        if pending_task is not None and not pending_task.done():
            pending_task.cancel()
            try:
                await pending_task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass


def _resolve_sdk_model() -> str | None:
    """Resolve the model name for the Claude Agent SDK CLI."""
    if config.claude_agent_model:
        return config.claude_agent_model
    if config.use_claude_code_subscription:
        return None
    model = config.model
    if "/" in model:
        model = model.split("/", 1)[1]
    if not config.openrouter_active:
        model = model.replace(".", "-")
    return model


def _build_sdk_env(
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, str]:
    """Build env vars for the SDK CLI subprocess."""
    if config.use_claude_code_subscription:
        _validate_claude_code_subscription()
        return {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_BASE_URL": "",
        }

    if not config.openrouter_active:
        return {}

    base = (config.base_url or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    env: dict[str, str] = {
        "ANTHROPIC_BASE_URL": base,
        "ANTHROPIC_AUTH_TOKEN": config.api_key or "",
        "ANTHROPIC_API_KEY": "",
    }

    def _safe(v: str) -> str:
        return v.replace("\r", "").replace("\n", "").strip()[:128]

    parts = []
    if session_id:
        parts.append(f"x-session-id: {_safe(session_id)}")
    if user_id:
        parts.append(f"x-user-id: {_safe(user_id)}")
    if parts:
        env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(parts)

    return env


def _make_sdk_cwd(session_id: str) -> str:
    """Create a safe, session-specific working directory path."""
    cwd = make_session_path(session_id)
    cwd = os.path.normpath(cwd)
    if not cwd.startswith(_SDK_CWD_PREFIX):
        raise ValueError(f"SDK cwd escaped prefix: {cwd}")
    return cwd


async def _cleanup_sdk_tool_results(cwd: str) -> None:
    """Remove SDK session artifacts for a specific working directory."""
    normalized = os.path.normpath(cwd)
    if not normalized.startswith(_SDK_CWD_PREFIX):
        logger.warning("[SDK] Rejecting cleanup for path outside workspace: %s", cwd)
        return

    await asyncio.to_thread(shutil.rmtree, normalized, True)

    global _last_sweep_time
    now = time.time()
    if now - _last_sweep_time >= _SWEEP_INTERVAL_SECONDS:
        _last_sweep_time = now
        encoded = encode_cwd_for_cli(normalized)
        await asyncio.to_thread(cleanup_stale_project_dirs, encoded)


def _format_sdk_content_blocks(blocks: list) -> list[dict[str, Any]]:
    """Convert SDK content blocks to transcript format."""
    result: list[dict[str, Any]] = []
    for block in blocks or []:
        if isinstance(block, TextBlock):
            result.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            result.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
        elif isinstance(block, ToolResultBlock):
            tool_result_entry: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
            }
            if block.is_error:
                tool_result_entry["is_error"] = True
            result.append(tool_result_entry)
        elif isinstance(block, ThinkingBlock):
            result.append(
                {
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                }
            )
        else:
            logger.warning(
                "[SDK] Unknown content block type: %s. "
                "This may indicate a new SDK version with additional block types.",
                type(block).__name__,
            )
    return result


async def _compress_messages(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], bool]:
    """Compress a list of messages if they exceed the token threshold."""
    messages = filter_compaction_messages(messages)

    if len(messages) < 2:
        return messages, False

    messages_dict = []
    for msg in messages:
        msg_dict: dict[str, Any] = {"role": msg.role}
        if msg.content:
            msg_dict["content"] = msg.content
        if msg.tool_calls:
            msg_dict["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id
        messages_dict.append(msg_dict)

    try:
        result = await _run_compression(messages_dict, config.model, "[SDK]")
    except Exception as exc:
        logger.warning("[SDK] _compress_messages failed, returning originals: %s", exc)
        return messages, False

    if result.was_compacted:
        logger.info(
            "[SDK] Context compacted: %s -> %s tokens (%s summarized, %s dropped)",
            result.original_token_count,
            result.token_count,
            result.messages_summarized,
            result.messages_dropped,
        )
        return [
            ChatMessage(
                role=m["role"],
                content=m.get("content"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in result.messages
        ], True

    return messages, False


def _format_conversation_context(messages: list[ChatMessage]) -> str | None:
    """Format conversation messages into a context prefix for the user message."""
    if not messages:
        return None

    messages = filter_compaction_messages(messages)

    lines: list[str] = []
    for msg in messages:
        if msg.role == "user":
            if msg.content:
                lines.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            if msg.content:
                lines.append(f"You responded: {msg.content}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "unknown")
                    tool_args = func.get("arguments", "")
                    lines.append(f"You called tool: {tool_name}({tool_args})")
        elif msg.role == "tool":
            content = msg.content or ""
            lines.append(f"Tool result: {content}")

    if not lines:
        return None

    return "<conversation_history>\n" + "\n".join(lines) + "\n</conversation_history>"


async def _build_query_message(
    current_message: str,
    session: ChatSession,
    use_resume: bool,
    transcript_msg_count: int,
    session_id: str,
) -> tuple[str, bool]:
    """Build the query message with appropriate context."""
    msg_count = len(session.messages)

    if use_resume and transcript_msg_count > 0:
        if transcript_msg_count < msg_count - 1:
            gap = session.messages[transcript_msg_count:-1]
            compressed, was_compressed = await _compress_messages(gap)
            gap_context = _format_conversation_context(compressed)
            if gap_context:
                logger.info(
                    "[SDK] Transcript stale: covers %d of %d messages, gap=%d (compressed=%s)",
                    transcript_msg_count,
                    msg_count,
                    len(gap),
                    was_compressed,
                )
                return (
                    f"{gap_context}\n\nNow, the user says:\n{current_message}",
                    was_compressed,
                )
    elif not use_resume and msg_count > 1:
        logger.warning(
            "[SDK] Using compression fallback for session %s (%d messages) — no transcript for --resume",
            session_id,
            msg_count,
        )
        compressed, was_compressed = await _compress_messages(session.messages[:-1])
        history_context = _format_conversation_context(compressed)
        if history_context:
            return (
                f"{history_context}\n\nNow, the user says:\n{current_message}",
                was_compressed,
            )

    return current_message, False


_VISION_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024
_UNSAFE_FILENAME = re.compile(r"[^\w.\-]")


def _save_to_sdk_cwd(sdk_cwd: str, filename: str, content: bytes) -> str:
    """Write file content to the SDK ephemeral directory."""
    safe = _UNSAFE_FILENAME.sub("_", filename) or "file"
    candidate = os.path.join(sdk_cwd, safe)
    if os.path.exists(candidate):
        stem, ext = os.path.splitext(safe)
        idx = 1
        while os.path.exists(candidate):
            candidate = os.path.join(sdk_cwd, f"{stem}_{idx}{ext}")
            idx += 1
    with open(candidate, "wb") as f:
        f.write(content)
    return candidate


class PreparedAttachments(BaseModel):
    """Result of preparing file attachments for a query."""

    hint: str = ""
    image_blocks: list[dict[str, Any]] = []


async def _prepare_file_attachments(
    file_ids: list[str],
    user_id: str,
    session_id: str,
    sdk_cwd: str,
) -> PreparedAttachments:
    """Download workspace files and prepare them for Claude."""
    empty = PreparedAttachments(hint="", image_blocks=[])
    if not file_ids or not user_id:
        return empty

    try:
        manager = await get_workspace_manager(user_id, session_id)
    except Exception:
        logger.warning(
            "Failed to create workspace manager for file attachments",
            exc_info=True,
        )
        return empty

    image_blocks: list[dict[str, Any]] = []
    file_descriptions: list[str] = []

    for fid in file_ids:
        try:
            file_info = await manager.get_file_info(fid)
            if file_info is None:
                continue
            content = await manager.read_file_by_id(fid)
            mime = (file_info.mime_type or "").split(";")[0].strip().lower()

            if mime in _VISION_MIME_TYPES and len(content) <= _MAX_INLINE_IMAGE_BYTES:
                b64 = base64.b64encode(content).decode("ascii")
                image_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    }
                )
                file_descriptions.append(
                    f"- {file_info.name} ({mime}, {file_info.size_bytes:,} bytes) [embedded as image]"
                )
            else:
                local_path = _save_to_sdk_cwd(sdk_cwd, file_info.name, content)
                file_descriptions.append(
                    f"- {file_info.name} ({mime}, {file_info.size_bytes:,} bytes) saved to {local_path}"
                )
        except Exception:
            logger.warning("Failed to prepare file %s", fid[:12], exc_info=True)

    if not file_descriptions:
        return empty

    noun = "file" if len(file_descriptions) == 1 else "files"
    has_non_images = len(file_descriptions) > len(image_blocks)
    read_hint = " Use the Read tool to view non-image files." if has_non_images else ""
    hint = (
        f"[The user attached {len(file_descriptions)} {noun}.{read_hint}\n"
        + "\n".join(file_descriptions)
        + "]"
    )
    return PreparedAttachments(hint=hint, image_blocks=image_blocks)


@dataclass
class _StreamAccumulator:
    """Mutable state accumulated during a single streaming attempt."""

    assistant_response: ChatMessage
    accumulated_tool_calls: list[dict[str, Any]]
    has_appended_assistant: bool = False
    has_tool_results: bool = False
    stream_completed: bool = False


def _dispatch_response(
    response: StreamBaseResponse,
    acc: _StreamAccumulator,
    ctx: "_StreamContext",
    state: "_RetryState",
    entries_replaced: bool,
    log_prefix: str,
) -> StreamBaseResponse | None:
    """Process a single adapter response and update session/accumulator state."""
    if isinstance(response, StreamStart):
        return None

    if isinstance(response, (StreamToolInputAvailable, StreamToolOutputAvailable)):
        extra = ""
        if isinstance(response, StreamToolOutputAvailable):
            out_len = len(str(response.output))
            extra = f", output_len={out_len}"
        logger.info(
            "%s Tool event: %s, tool=%s%s",
            log_prefix,
            type(response).__name__,
            getattr(response, "toolName", "N/A"),
            extra,
        )

    if isinstance(response, StreamError):
        logger.error(
            "%s Sending error to frontend: %s (code=%s)",
            log_prefix,
            response.errorText,
            response.code,
        )
        _append_error_marker(
            ctx.session,
            response.errorText,
            retryable=(response.code == "transient_api_error"),
        )

    if isinstance(response, StreamTextDelta):
        delta = response.delta or ""
        if acc.has_tool_results and acc.has_appended_assistant:
            acc.assistant_response.content = (
                acc.assistant_response.content or ""
            ) + delta
            acc.has_tool_results = False
        else:
            acc.assistant_response.content = (
                acc.assistant_response.content or ""
            ) + delta
            if not acc.has_appended_assistant:
                ctx.session.messages.append(acc.assistant_response)
                acc.has_appended_assistant = True

    elif isinstance(response, StreamToolInputAvailable):
        acc.accumulated_tool_calls.append(
            {
                "id": response.toolCallId,
                "type": "function",
                "function": {
                    "name": response.toolName,
                    "arguments": json.dumps(response.input or {}),
                },
            }
        )
        acc.assistant_response.tool_calls = acc.accumulated_tool_calls
        if not acc.has_appended_assistant:
            ctx.session.messages.append(acc.assistant_response)
            acc.has_appended_assistant = True

    elif isinstance(response, StreamToolOutputAvailable):
        content = (
            response.output
            if isinstance(response.output, str)
            else json.dumps(response.output, ensure_ascii=False)
        )
        ctx.session.messages.append(
            ChatMessage(
                role="tool",
                content=content,
                tool_call_id=response.toolCallId,
            )
        )
        if not entries_replaced:
            state.transcript_builder.append_tool_result(
                tool_use_id=response.toolCallId,
                content=content,
            )
        acc.has_tool_results = True

    elif isinstance(response, StreamFinish):
        acc.stream_completed = True

    return response


class _HandledStreamError(Exception):
    """Raised by `_run_stream_attempt` after it has already yielded a `StreamError`."""

    def __init__(
        self,
        message: str,
        error_msg: str | None = None,
        code: str | None = None,
        retryable: bool = True,
    ):
        super().__init__(message)
        self.error_msg = error_msg
        self.code = code
        self.retryable = retryable


@dataclass
class _EmptyToolBreakResult:
    """Result of checking for empty tool calls in a single AssistantMessage."""

    count: int
    tripped: bool
    error: StreamError | None
    error_msg: str | None
    error_code: str | None


def _check_empty_tool_breaker(
    sdk_msg: object,
    consecutive: int,
    ctx: _StreamContext,
    state: _RetryState,
) -> _EmptyToolBreakResult:
    """Detect consecutive empty tool calls and trip the circuit breaker."""
    if not isinstance(sdk_msg, AssistantMessage):
        return _EmptyToolBreakResult(consecutive, False, None, None, None)

    empty_tools = [
        b.name for b in sdk_msg.content if isinstance(b, ToolUseBlock) and not b.input
    ]
    if not empty_tools:
        return _EmptyToolBreakResult(0, False, None, None, None)

    consecutive += 1

    if consecutive == 1:
        logger.warning(
            "%s Empty tool call detected (%d/%d): tools=%s, model=%s, error=%s, block_types=%s, cumulative_usage=%s",
            ctx.log_prefix,
            consecutive,
            _EMPTY_TOOL_CALL_LIMIT,
            empty_tools,
            sdk_msg.model,
            sdk_msg.error,
            [type(b).__name__ for b in sdk_msg.content],
            {
                "prompt": state.usage.prompt_tokens,
                "completion": state.usage.completion_tokens,
                "cache_read": state.usage.cache_read_tokens,
            },
        )
    else:
        logger.warning(
            "%s Empty tool call detected (%d/%d): tools=%s",
            ctx.log_prefix,
            consecutive,
            _EMPTY_TOOL_CALL_LIMIT,
            empty_tools,
        )

    if consecutive < _EMPTY_TOOL_CALL_LIMIT:
        return _EmptyToolBreakResult(consecutive, False, None, None, None)

    logger.error(
        "%s Circuit breaker: aborting stream after %d consecutive empty tool calls. "
        "This is likely caused by the model attempting to write content too large "
        "for a single tool call's output token limit. The model should write large "
        "files in chunks using bash_exec with cat >> (append).",
        ctx.log_prefix,
        consecutive,
    )
    error_msg = _CIRCUIT_BREAKER_ERROR_MSG
    error_code = "circuit_breaker_empty_tool_calls"
    _append_error_marker(ctx.session, error_msg, retryable=True)
    return _EmptyToolBreakResult(
        count=consecutive,
        tripped=True,
        error=StreamError(errorText=error_msg, code=error_code),
        error_msg=error_msg,
        error_code=error_code,
    )


async def _run_stream_attempt(
    ctx: _StreamContext,
    state: _RetryState,
) -> AsyncIterator[StreamBaseResponse]:
    """Run one SDK streaming attempt."""
    acc = _StreamAccumulator(
        assistant_response=ChatMessage(
            id=ctx.message_id,
            role="assistant",
            content="",
        ),
        accumulated_tool_calls=[],
    )
    ended_with_stream_error = False
    stream_error_msg: str | None = None
    stream_error_code: str | None = None

    consecutive_empty_tool_calls = 0

    async with ClaudeSDKClient(options=state.options) as client:
        logger.info(
            "%s Sending query — resume=%s, total_msgs=%d, query_len=%d, attached_files=%d, image_blocks=%d",
            ctx.log_prefix,
            state.use_resume,
            len(ctx.session.messages),
            len(state.query_message),
            len(ctx.file_ids) if ctx.file_ids else 0,
            len(ctx.attachments.image_blocks),
        )

        ctx.compaction.reset_for_query()
        if state.was_compacted:
            for ev in ctx.compaction.emit_pre_query(ctx.session):
                yield ev

        if ctx.attachments.image_blocks:
            content_blocks: list[dict[str, Any]] = [
                *ctx.attachments.image_blocks,
                {"type": "text", "text": state.query_message},
            ]
            user_msg = {
                "type": "user",
                "message": {"role": "user", "content": content_blocks},
                "parent_tool_use_id": None,
                "session_id": ctx.session_id,
            }
            if client._transport is None:  # noqa: SLF001
                raise RuntimeError("ClaudeSDKClient transport is not initialized")
            await client._transport.write(json.dumps(user_msg) + "\n")  # noqa: SLF001
            state.transcript_builder.append_user(
                content=[
                    *ctx.attachments.image_blocks,
                    {"type": "text", "text": ctx.current_message},
                ]
            )
        else:
            await client.query(state.query_message, session_id=ctx.session_id)
            state.transcript_builder.append_user(content=ctx.current_message)

        async for sdk_msg in _iter_sdk_messages(client):
            if sdk_msg is None:
                await ctx.lock.refresh()
                for ev in ctx.compaction.emit_start_if_ready():
                    yield ev
                yield StreamHeartbeat()
                continue

            logger.info(
                "%s Received: %s %s (unresolved=%d, current=%d, resolved=%d)",
                ctx.log_prefix,
                type(sdk_msg).__name__,
                getattr(sdk_msg, "subtype", ""),
                len(state.adapter.current_tool_calls)
                - len(state.adapter.resolved_tool_calls),
                len(state.adapter.current_tool_calls),
                len(state.adapter.resolved_tool_calls),
            )

            sdk_error = getattr(sdk_msg, "error", None)
            if isinstance(sdk_msg, AssistantMessage) and sdk_error:
                error_text = str(sdk_error)
                error_preview = str(sdk_msg.content)[:500]
                logger.error(
                    "[SDK] [%s] AssistantMessage has error=%s, content_blocks=%d, content_preview=%s",
                    ctx.session_id[:12],
                    sdk_error,
                    len(sdk_msg.content),
                    error_preview,
                )

                if is_transient_api_error(error_text) or is_transient_api_error(
                    error_preview
                ):
                    logger.warning(
                        "%s Transient Anthropic API error detected, suppressing raw error text",
                        ctx.log_prefix,
                    )
                    stream_error_msg = FRIENDLY_TRANSIENT_MSG
                    stream_error_code = "transient_api_error"
                    _append_error_marker(
                        ctx.session,
                        stream_error_msg,
                        retryable=True,
                    )
                    yield StreamError(
                        errorText=stream_error_msg,
                        code=stream_error_code,
                    )
                    ended_with_stream_error = True
                    break

            is_tool_only = False
            if isinstance(sdk_msg, AssistantMessage) and sdk_msg.content:
                is_tool_only = True
                for tool_use in sdk_msg.content:
                    if isinstance(tool_use, ToolUseBlock):
                        await pre_launch_tool_call(tool_use.name, tool_use.input)
                    else:
                        is_tool_only = False

            if (
                state.adapter.has_unresolved_tool_calls
                and isinstance(sdk_msg, (AssistantMessage, ResultMessage))
                and not is_tool_only
            ):
                if await wait_for_stash():
                    await asyncio.sleep(0)
                else:
                    logger.warning(
                        "%s Timed out waiting for PostToolUse hook stash (%d unresolved tool calls)",
                        ctx.log_prefix,
                        len(state.adapter.current_tool_calls)
                        - len(state.adapter.resolved_tool_calls),
                    )

            if isinstance(sdk_msg, ResultMessage):
                logger.info(
                    "%s Received: ResultMessage %s (unresolved=%d, current=%d, resolved=%d, num_turns=%d, cost_usd=%s, result=%s)",
                    ctx.log_prefix,
                    sdk_msg.subtype,
                    len(state.adapter.current_tool_calls)
                    - len(state.adapter.resolved_tool_calls),
                    len(state.adapter.current_tool_calls),
                    len(state.adapter.resolved_tool_calls),
                    sdk_msg.num_turns,
                    sdk_msg.total_cost_usd,
                    (sdk_msg.result or "")[:200],
                )
                if sdk_msg.subtype in ("error", "error_during_execution"):
                    logger.error(
                        "%s SDK execution failed with error: %s",
                        ctx.log_prefix,
                        sdk_msg.result or "(no error message provided)",
                    )

                if sdk_msg.usage:
                    state.usage.prompt_tokens += sdk_msg.usage.get("input_tokens", 0)
                    state.usage.cache_read_tokens += sdk_msg.usage.get(
                        "cache_read_input_tokens", 0
                    )
                    state.usage.cache_creation_tokens += sdk_msg.usage.get(
                        "cache_creation_input_tokens", 0
                    )
                    state.usage.completion_tokens += sdk_msg.usage.get(
                        "output_tokens", 0
                    )
                    logger.info(
                        "%s Token usage: uncached=%d, cache_read=%d, cache_create=%d, output=%d",
                        ctx.log_prefix,
                        state.usage.prompt_tokens,
                        state.usage.cache_read_tokens,
                        state.usage.cache_creation_tokens,
                        state.usage.completion_tokens,
                    )
                if sdk_msg.total_cost_usd is not None:
                    state.usage.cost_usd = sdk_msg.total_cost_usd

            compact_result = await ctx.compaction.emit_end_if_ready(ctx.session)
            for ev in compact_result.events:
                yield ev
            entries_replaced = False
            if compact_result.just_ended:
                compacted = await asyncio.to_thread(
                    read_compacted_entries,
                    compact_result.transcript_path,
                )
                if compacted is not None:
                    state.transcript_builder.replace_entries(
                        compacted, log_prefix=ctx.log_prefix
                    )
                    entries_replaced = True

            breaker = _check_empty_tool_breaker(
                sdk_msg, consecutive_empty_tool_calls, ctx, state
            )
            consecutive_empty_tool_calls = breaker.count
            if breaker.tripped and breaker.error is not None:
                stream_error_msg = breaker.error_msg
                stream_error_code = breaker.error_code
                yield breaker.error
                ended_with_stream_error = True
                break

            for response in state.adapter.convert_message(sdk_msg):
                dispatched = _dispatch_response(
                    response, acc, ctx, state, entries_replaced, ctx.log_prefix
                )
                if dispatched is not None:
                    yield dispatched

            if isinstance(sdk_msg, AssistantMessage) and not entries_replaced:
                state.transcript_builder.append_assistant(
                    content_blocks=_format_sdk_content_blocks(sdk_msg.content),
                    model=sdk_msg.model,
                )

            if acc.stream_completed:
                break

    if state.adapter.has_unresolved_tool_calls:
        logger.warning(
            "%s %d unresolved tool(s) after stream — flushing",
            ctx.log_prefix,
            len(state.adapter.current_tool_calls)
            - len(state.adapter.resolved_tool_calls),
        )
        safety_responses: list[StreamBaseResponse] = []
        state.adapter._flush_unresolved_tool_calls(safety_responses)
        for response in safety_responses:
            if isinstance(
                response,
                (StreamToolInputAvailable, StreamToolOutputAvailable),
            ):
                logger.info(
                    "%s Safety flush: %s, tool=%s",
                    ctx.log_prefix,
                    type(response).__name__,
                    getattr(response, "toolName", "N/A"),
                )
            if isinstance(response, StreamToolOutputAvailable):
                state.transcript_builder.append_tool_result(
                    tool_use_id=response.toolCallId,
                    content=(
                        response.output
                        if isinstance(response.output, str)
                        else json.dumps(response.output, ensure_ascii=False)
                    ),
                )
            yield response

    if not acc.stream_completed and not ended_with_stream_error:
        logger.info(
            "%s Stream ended without ResultMessage (stopped by user)",
            ctx.log_prefix,
        )
        closing_responses: list[StreamBaseResponse] = []
        state.adapter._end_text_if_open(closing_responses)
        for r in closing_responses:
            yield r
        ctx.session.messages.append(
            ChatMessage(
                role="assistant",
                content=f"{COPILOT_SYSTEM_PREFIX} Execution stopped by user",
            )
        )

    if (
        acc.assistant_response.content or acc.assistant_response.tool_calls
    ) and not acc.has_appended_assistant:
        ctx.session.messages.append(acc.assistant_response)

    if ended_with_stream_error:
        raise _HandledStreamError(
            "Stream error handled — StreamError already yielded",
            error_msg=stream_error_msg,
            code=stream_error_code,
        )


async def stream_chat_completion_sdk(
    session_id: str,
    message: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    session: ChatSession | None = None,
    file_ids: list[str] | None = None,
    permissions: "CopilotPermissions | None" = None,
    **_kwargs: Any,
) -> AsyncIterator[StreamBaseResponse]:
    """Stream chat completion using Claude Agent SDK."""
    if session is None:
        session = await get_chat_session(session_id, user_id)

    if not session:
        raise NotFoundError(
            f"Session {session_id} not found. Please create a new session first."
        )

    session = cast(ChatSession, session)

    while (
        len(session.messages) > 0
        and session.messages[-1].role == "assistant"
        and session.messages[-1].content
        and (
            COPILOT_ERROR_PREFIX in session.messages[-1].content
            or COPILOT_RETRYABLE_ERROR_PREFIX in session.messages[-1].content
        )
    ):
        logger.info(
            "[SDK] [%s] Removing stale error marker from previous turn",
            session_id[:12],
        )
        session.messages.pop()

    new_message_role = "user" if is_user_message else "assistant"
    if message and (
        len(session.messages) == 0
        or not (
            session.messages[-1].role == new_message_role
            and session.messages[-1].content == message
        )
    ):
        session.messages.append(ChatMessage(role=new_message_role, content=message))
        if is_user_message:
            track_user_message(
                user_id=user_id, session_id=session_id, message_length=len(message)
            )

    turn = sum(1 for m in session.messages if m.role == "user")
    log_prefix = f"[SDK][{session_id[:12]}][T{turn}]"

    session = await upsert_chat_session(session)

    if is_user_message and not session.title:
        user_messages = [m for m in session.messages if m.role == "user"]
        if len(user_messages) == 1:
            first_message = user_messages[0].content or message or ""
            if first_message:
                task = asyncio.create_task(
                    _update_title_async(session_id, first_message, user_id)
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

    message_id = str(uuid.uuid4())
    stream_id = str(uuid.uuid4())
    ended_with_stream_error = False
    e2b_sandbox = None
    use_resume = False
    resume_file: str | None = None
    transcript_builder = TranscriptBuilder()
    sdk_cwd = ""
    transcript_covers_prefix = True

    lock = AsyncClusterLock(
        redis=await get_redis_async(),
        key=f"{STREAM_LOCK_PREFIX}{session_id}",
        owner_id=stream_id,
        timeout=config.stream_lock_ttl,
    )

    lock_owner = await lock.try_acquire()
    if lock_owner != stream_id:
        logger.warning(
            "%s Session already has an active stream: %s", log_prefix, lock_owner
        )
        yield StreamError(
            errorText="Another stream is already active for this session. Please wait or stop it.",
            code="stream_already_active",
        )
        return

    _otel_ctx: Any = None
    skip_transcript_upload = False
    transcript_content: str = ""
    state: _RetryState | None = None

    turn_prompt_tokens = 0
    turn_completion_tokens = 0
    turn_cache_read_tokens = 0
    turn_cache_creation_tokens = 0
    turn_cost_usd: float | None = None

    try:
        has_history = len(session.messages) > 1
        try:
            sdk_cwd = _make_sdk_cwd(session_id)
            os.makedirs(sdk_cwd, exist_ok=True)
        except (ValueError, OSError) as e:
            logger.error("%s Invalid SDK cwd: %s", log_prefix, e)
            yield StreamError(
                errorText="Unable to initialize working directory.",
                code="sdk_cwd_error",
            )
            return

        async def _setup_e2b():
            if not (e2b_api_key := config.active_e2b_api_key):
                if config.use_e2b_sandbox:
                    logger.warning(
                        "[E2B] [%s] E2B sandbox enabled but no API key configured (CHAT_E2B_API_KEY / E2B_API_KEY) — falling back to bubblewrap",
                        session_id[:12],
                    )
                return None
            try:
                sandbox = await get_or_create_sandbox(
                    session_id,
                    api_key=e2b_api_key,
                    template=config.e2b_sandbox_template,
                    timeout=config.e2b_sandbox_timeout,
                    on_timeout=config.e2b_sandbox_on_timeout,
                )
            except Exception as e2b_err:
                logger.error(
                    "[E2B] [%s] Setup failed: %s",
                    session_id[:12],
                    e2b_err,
                    exc_info=True,
                )
                return None
            return sandbox

        async def _fetch_transcript():
            if not (
                config.claude_agent_use_resume and user_id and len(session.messages) > 1
            ):
                return None
            try:
                return await download_transcript(
                    user_id, session_id, log_prefix=log_prefix
                )
            except Exception as transcript_err:
                logger.warning(
                    "%s Transcript download failed, continuing without --resume: %s",
                    log_prefix,
                    transcript_err,
                )
                return None

        e2b_sandbox, (base_system_prompt, _), dl = await asyncio.gather(
            _setup_e2b(),
            _build_system_prompt(user_id, has_conversation_history=has_history),
            _fetch_transcript(),
        )

        use_e2b = e2b_sandbox is not None
        system_prompt = base_system_prompt + get_sdk_supplement(
            use_e2b=use_e2b, cwd=sdk_cwd
        )

        transcript_msg_count = 0
        if dl:
            is_valid = validate_transcript(dl.content)
            dl_lines = dl.content.strip().split("\n") if dl.content else []
            logger.info(
                "%s Downloaded transcript: %dB, %d lines, msg_count=%d, valid=%s",
                log_prefix,
                len(dl.content),
                len(dl_lines),
                dl.message_count,
                is_valid,
            )
            if is_valid:
                transcript_content = dl.content
                transcript_builder.load_previous(dl.content, log_prefix=log_prefix)
                resume_file = await asyncio.to_thread(
                    write_transcript_to_tempfile, dl.content, session_id, sdk_cwd
                )
                if resume_file:
                    use_resume = True
                    transcript_msg_count = dl.message_count
                    logger.debug(
                        "%s Using --resume (%dB, msg_count=%d)",
                        log_prefix,
                        len(dl.content),
                        transcript_msg_count,
                    )
            else:
                logger.warning("%s Transcript downloaded but invalid", log_prefix)
                transcript_covers_prefix = False
        elif config.claude_agent_use_resume and user_id and len(session.messages) > 1:
            logger.warning(
                "%s No transcript available (%d messages in session)",
                log_prefix,
                len(session.messages),
            )
            transcript_covers_prefix = False

        yield StreamStart(messageId=message_id, sessionId=session_id)

        set_execution_context(
            user_id,
            session,
            sandbox=e2b_sandbox,
            sdk_cwd=sdk_cwd,
            permissions=permissions,
        )

        sdk_env = _build_sdk_env(session_id=session_id, user_id=user_id)
        if not config.api_key and not config.use_claude_code_subscription:
            raise RuntimeError(
                "No API key configured. Set OPEN_ROUTER_API_KEY, CHAT_API_KEY, or ANTHROPIC_API_KEY for API access, "
                "or CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true to use Claude Code CLI subscription (requires `claude login`)."
            )

        mcp_server = create_copilot_mcp_server(use_e2b=use_e2b)

        sdk_model = _resolve_sdk_model()
        compaction = CompactionTracker()

        security_hooks = create_security_hooks(
            user_id,
            sdk_cwd=sdk_cwd,
            max_subtasks=config.claude_agent_max_subtasks,
            on_compact=compaction.on_compact,
        )

        if permissions is not None:
            allowed, disallowed = apply_tool_permissions(permissions, use_e2b=use_e2b)
        else:
            allowed = get_copilot_tool_names(use_e2b=use_e2b)
            disallowed = get_sdk_disallowed_tools(use_e2b=use_e2b)

        def _on_stderr(line: str) -> None:
            sid = session_id[:12] if session_id else "?"
            logger.info("[SDK] [%s] CLI stderr: %s", sid, line.rstrip())

        sdk_options_kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "mcp_servers": {"copilot": mcp_server},
            "allowed_tools": allowed,
            "disallowed_tools": disallowed,
            "hooks": security_hooks,
            "cwd": sdk_cwd,
            "max_buffer_size": config.claude_agent_max_buffer_size,
            "stderr": _on_stderr,
        }
        if sdk_model:
            sdk_options_kwargs["model"] = sdk_model
        if sdk_env:
            sdk_options_kwargs["env"] = sdk_env
        if use_resume and resume_file:
            sdk_options_kwargs["resume"] = resume_file

        options = ClaudeAgentOptions(**sdk_options_kwargs)  # type: ignore[arg-type]

        adapter = SDKResponseAdapter(message_id=message_id, session_id=session_id)

        _otel_ctx = propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            trace_name="copilot-sdk",
            tags=["sdk"],
            metadata={
                "resume": str(use_resume),
                "conversation_turn": str(turn),
            },
        )
        _otel_ctx.__enter__()

        current_message = message or ""
        if not current_message and session.messages:
            last_user = [m for m in session.messages if m.role == "user"]
            if last_user:
                current_message = last_user[-1].content or ""

        if not current_message.strip():
            yield StreamError(
                errorText="Message cannot be empty.",
                code="empty_prompt",
            )
            return

        query_message, was_compacted = await _build_query_message(
            current_message,
            session,
            use_resume,
            transcript_msg_count,
            session_id,
        )

        attachments = await _prepare_file_attachments(
            file_ids or [], user_id or "", session_id, sdk_cwd
        )
        if attachments.hint:
            query_message = f"{query_message}\n\n{attachments.hint}"

        tried_compaction = False

        stream_ctx = _StreamContext(
            session=session,
            session_id=session_id,
            log_prefix=log_prefix,
            sdk_cwd=sdk_cwd,
            current_message=current_message,
            file_ids=file_ids,
            message_id=message_id,
            attachments=attachments,
            compaction=compaction,
            lock=lock,
        )

        ended_with_stream_error = False
        attempts_exhausted = False
        stream_err: Exception | None = None

        state = _RetryState(
            options=options,
            query_message=query_message,
            was_compacted=was_compacted,
            use_resume=use_resume,
            resume_file=resume_file,
            transcript_msg_count=transcript_msg_count,
            adapter=adapter,
            transcript_builder=transcript_builder,
            usage=_TokenUsage(),
        )

        attempt = 0
        while attempt < _MAX_STREAM_ATTEMPTS:
            current_message_id = (
                message_id if attempt == 0 else str(uuid.uuid4())
            )
            stream_ctx = replace(stream_ctx, message_id=current_message_id)
            state.adapter = SDKResponseAdapter(
                message_id=current_message_id,
                session_id=session_id,
            )
            state.usage.reset()

            reset_stash_event()
            reset_tool_failure_counters()

            if attempt > 0:
                logger.info(
                    "%s Retrying with reduced context (%d/%d)",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                )
                yield StreamStatus(message="Optimizing conversation context…")

                ctx = await _reduce_context(
                    transcript_content,
                    tried_compaction,
                    session_id,
                    sdk_cwd,
                    log_prefix,
                )
                state.transcript_builder = ctx.builder
                state.use_resume = ctx.use_resume
                state.resume_file = ctx.resume_file
                tried_compaction = ctx.tried_compaction
                state.transcript_msg_count = 0
                if ctx.transcript_lost:
                    skip_transcript_upload = True

                sdk_options_kwargs_retry = dict(sdk_options_kwargs)
                if ctx.use_resume and ctx.resume_file:
                    sdk_options_kwargs_retry["resume"] = ctx.resume_file
                elif "resume" in sdk_options_kwargs_retry:
                    del sdk_options_kwargs_retry["resume"]

                state.options = ClaudeAgentOptions(**sdk_options_kwargs_retry)  # type: ignore[arg-type]
                state.query_message, state.was_compacted = await _build_query_message(
                    current_message,
                    session,
                    state.use_resume,
                    state.transcript_msg_count,
                    session_id,
                )
                if attachments.hint:
                    state.query_message = f"{state.query_message}\n\n{attachments.hint}"

            pre_attempt_msg_count = len(session.messages)
            events_yielded = 0

            try:
                async for event in _run_stream_attempt(stream_ctx, state):
                    if not isinstance(event, StreamHeartbeat):
                        events_yielded += 1
                    yield event
                await cancel_pending_tool_tasks()
                break
            except asyncio.CancelledError:
                logger.warning(
                    "%s Streaming cancelled (attempt %d/%d)",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                )
                await cancel_pending_tool_tasks()
                raise
            except _HandledStreamError as exc:
                logger.warning(
                    "%s Stream error handled in attempt (attempt %d/%d, code=%s, events_yielded=%d)",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                    exc.code or "transient",
                    events_yielded,
                )
                session.messages = session.messages[:pre_attempt_msg_count]
                skip_transcript_upload = True
                _append_error_marker(
                    session,
                    exc.error_msg or FRIENDLY_TRANSIENT_MSG,
                    retryable=True,
                )
                ended_with_stream_error = True
                await cancel_pending_tool_tasks()
                break
            except Exception as e:
                stream_err = e
                is_context_error = _is_prompt_too_long(e)
                logger.warning(
                    "%s Stream error (attempt %d/%d, context_error=%s, events_yielded=%d): %s",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                    is_context_error,
                    events_yielded,
                    stream_err,
                    exc_info=True,
                )
                session.messages = session.messages[:pre_attempt_msg_count]
                await cancel_pending_tool_tasks()
                if events_yielded > 0:
                    logger.warning(
                        "%s Not retrying — %d events already yielded",
                        log_prefix,
                        events_yielded,
                    )
                    skip_transcript_upload = True
                    ended_with_stream_error = True
                    break
                if not is_context_error:
                    skip_transcript_upload = True
                    ended_with_stream_error = True
                    break
                attempt += 1
                continue
        else:
            ended_with_stream_error = True
            attempts_exhausted = True
            logger.error(
                "%s All %d query attempts exhausted: %s",
                log_prefix,
                _MAX_STREAM_ATTEMPTS,
                stream_err,
            )

        if ended_with_stream_error and state is not None:
            error_flush: list[StreamBaseResponse] = []
            state.adapter._end_text_if_open(error_flush)
            if state.adapter.has_unresolved_tool_calls:
                logger.warning(
                    "%s Flushing %d unresolved tool(s) after stream error",
                    log_prefix,
                    len(state.adapter.current_tool_calls)
                    - len(state.adapter.resolved_tool_calls),
                )
                state.adapter._flush_unresolved_tool_calls(error_flush)
            for response in error_flush:
                yield response

        if ended_with_stream_error and stream_err is not None:
            safe_err = str(stream_err).replace("\n", " ").replace("\r", "")[:500]
            if attempts_exhausted:
                error_text = (
                    "Your conversation is too long. "
                    "Please start a new chat or clear some history."
                )
            else:
                error_text = _friendly_error_text(safe_err)
            yield StreamError(
                errorText=error_text,
                code=(
                    "all_attempts_exhausted"
                    if attempts_exhausted
                    else "sdk_stream_error"
                ),
            )

        if state is not None:
            turn_prompt_tokens = state.usage.prompt_tokens
            turn_completion_tokens = state.usage.completion_tokens
            turn_cache_read_tokens = state.usage.cache_read_tokens
            turn_cache_creation_tokens = state.usage.cache_creation_tokens
            turn_cost_usd = state.usage.cost_usd

        if turn_prompt_tokens > 0 or turn_completion_tokens > 0:
            total_tokens = turn_prompt_tokens + turn_completion_tokens
            yield StreamUsage(
                prompt_tokens=turn_prompt_tokens,
                completion_tokens=turn_completion_tokens,
                total_tokens=total_tokens,
                cache_read_tokens=turn_cache_read_tokens,
                cache_creation_tokens=turn_cache_creation_tokens,
            )

        if ended_with_stream_error:
            logger.warning(
                "%s Stream ended with SDK error after %d messages",
                log_prefix,
                len(session.messages),
            )
        else:
            logger.info(
                "%s Stream completed successfully with %d messages",
                log_prefix,
                len(session.messages),
            )
    except GeneratorExit:
        logger.warning("%s GeneratorExit — releasing stream lock", log_prefix)
        await lock.release()
        raise
    except BaseException as e:
        if isinstance(e, asyncio.CancelledError):
            logger.warning("%s Session cancelled", log_prefix)
            error_msg = "Operation cancelled"
        else:
            error_msg = str(e) or type(e).__name__
            if isinstance(e, RuntimeError) and "cancel scope" in str(e):
                logger.warning("%s SDK cleanup error: %s", log_prefix, error_msg)
            else:
                logger.error("%s Error: %s", log_prefix, error_msg, exc_info=True)

        is_transient = is_transient_api_error(error_msg)
        if is_transient:
            display_msg, code = FRIENDLY_TRANSIENT_MSG, "transient_api_error"
        else:
            display_msg, code = error_msg, "sdk_error"

        if not ended_with_stream_error:
            _append_error_marker(session, display_msg, retryable=is_transient)
            logger.debug(
                "%s Appended error marker, will be persisted in finally",
                log_prefix,
            )

        is_cancellation = isinstance(e, asyncio.CancelledError) or (
            isinstance(e, RuntimeError) and "cancel scope" in str(e)
        )
        if not is_cancellation:
            yield StreamError(errorText=display_msg, code=code)

        raise
    finally:
        if _otel_ctx is not None:
            try:
                _otel_ctx.__exit__(*sys.exc_info())
            except Exception:
                logger.warning("OTEL context teardown failed", exc_info=True)

        await persist_and_record_usage(
            session=session,
            user_id=user_id,
            prompt_tokens=turn_prompt_tokens,
            completion_tokens=turn_completion_tokens,
            cache_read_tokens=turn_cache_read_tokens,
            cache_creation_tokens=turn_cache_creation_tokens,
            log_prefix=log_prefix,
            cost_usd=turn_cost_usd,
        )

        if session is not None:
            try:
                await asyncio.shield(upsert_chat_session(session))
                logger.info(
                    "%s Session persisted in finally with %d messages",
                    log_prefix,
                    len(session.messages),
                )
            except Exception as persist_err:
                logger.error(
                    "%s Failed to persist session in finally: %s",
                    log_prefix,
                    persist_err,
                    exc_info=True,
                )

        if e2b_sandbox is not None:
            task = asyncio.create_task(pause_sandbox_direct(e2b_sandbox, session_id))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

        if skip_transcript_upload:
            logger.warning(
                "%s Skipping transcript upload — transcript was dropped during prompt-too-long recovery",
                log_prefix,
            )
        elif (
            config.claude_agent_use_resume
            and user_id
            and session is not None
            and state is not None
        ):
            try:
                transcript_upload_content = state.transcript_builder.to_jsonl()
                entry_count = state.transcript_builder.entry_count

                if not transcript_upload_content:
                    logger.warning(
                        "%s No transcript to upload (builder empty)", log_prefix
                    )
                elif not validate_transcript(transcript_upload_content):
                    logger.warning(
                        "%s Transcript invalid, skipping upload (entries=%d)",
                        log_prefix,
                        entry_count,
                    )
                elif not transcript_covers_prefix:
                    logger.warning(
                        "%s Skipping transcript upload — builder does not cover full session prefix (entries=%d, session=%d)",
                        log_prefix,
                        entry_count,
                        len(session.messages),
                    )
                else:
                    logger.info(
                        "%s Uploading transcript (entries=%d, bytes=%d)",
                        log_prefix,
                        entry_count,
                        len(transcript_upload_content),
                    )
                    await asyncio.shield(
                        upload_transcript(
                            user_id=user_id,
                            session_id=session_id,
                            content=transcript_upload_content,
                            message_count=len(session.messages),
                            log_prefix=log_prefix,
                        )
                    )
            except Exception as upload_err:
                logger.error(
                    "%s Transcript upload failed in finally: %s",
                    log_prefix,
                    upload_err,
                    exc_info=True,
                )

        try:
            if sdk_cwd:
                await _cleanup_sdk_tool_results(sdk_cwd)
        except Exception:
            logger.warning("%s SDK cleanup failed", log_prefix, exc_info=True)
        finally:
            await lock.release()


async def _update_title_async(
    session_id: str, message: str, user_id: str | None = None
) -> None:
    """Background task to update session title."""
    try:
        title = await _generate_session_title(
            message, user_id=user_id, session_id=session_id
        )
        if title and user_id:
            await update_session_title(session_id, user_id, title, only_if_empty=True)
            logger.debug("[SDK] Generated title for %s: %s", session_id, title)
    except Exception as e:
        logger.warning("[SDK] Failed to update session title: %s", e)
