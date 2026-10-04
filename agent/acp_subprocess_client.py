"""Generic persistent-session ACP subprocess client (OpenAI-shim).

Unlike ``CopilotACPClient`` — which spawns a fresh ``copilot --acp`` process,
crams the whole conversation into one prompt, and kills the process after every
message — this client keeps ONE adapter process and ONE ACP session alive across
turns of a Hermes session, and sends only the *delta* (the new user turn) each
call. That matches stateful ACP agents like ``@agentclientprotocol/claude-agent-acp``,
whose session accumulates context on the adapter side; replaying the full history
every turn would duplicate context (double token/credit spend).

Design notes
------------
* The instance is 1:1 with a Hermes session (it lives on ``agent.client``). The
  persistent process/session is created lazily on the first ``create()`` and torn
  down by ``close()`` (session end / idle / provider switch).
* ``claude-agent-acp`` runs its OWN tools (Read/Write/Bash …) internally; it does
  NOT delegate file ops to the client via ``fs/*`` and (in ``bypassPermissions``
  mode) does NOT ask the client for permission. So this shim does NOT inject
  Hermes tool schemas and does NOT parse ``<tool_call>`` blocks — each ``create()``
  is one user turn and yields one final assistant text answer.
* Cost is taken straight from the adapter's ``usage_update`` notification
  (``cost.amount`` in USD, cumulative per session); we expose the per-turn delta.
* Permission posture is configurable (default: ``bypassPermissions`` — the
  operator opted into adapter-side execution with Hermes restricting via cwd/env).

Stage references map to scripts/claude-agent-acp-hermes-plan-v4.md.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Reuse the proven helpers from the Copilot shim rather than duplicating them.
from agent.copilot_acp_client import (
    _build_subprocess_env,
    _render_message_content,
)

ACP_MARKER_BASE_URL = "acp://claude-agent"
DEFAULT_COMMAND = "claude-agent-acp"
_DEFAULT_TIMEOUT_SECONDS = 900.0
# Default permission posture for the adapter's own tool calls. The adapter
# executes within the configured cwd/env. Override with the
# HERMES_CLAUDE_AGENT_ACP_PERMISSION_MODE environment variable.
DEFAULT_PERMISSION_MODE = "bypassPermissions"
_PROTOCOL_VERSION = 1

# Map ACP PromptResponse.stopReason → OpenAI finish_reason.
_ACP_STOP_REASON_MAP = {
    "completed": "stop",
    "end_turn": "stop",
    "cancelled": "stop",
    "canceled": "stop",
    "max_tokens": "length",
    "maxtokens": "length",
    "max_turn_requests": "length",
    "refusal": "stop",
}
logger = logging.getLogger(__name__)


def _resolve_command(command: str | None) -> str:
    return (
        (command or "").strip()
        or os.getenv("HERMES_CLAUDE_AGENT_ACP_COMMAND", "").strip()
        or DEFAULT_COMMAND
    )


def _resolve_permission_mode(explicit: str | None) -> str:
    return (
        (explicit or "").strip()
        or os.getenv("HERMES_CLAUDE_AGENT_ACP_PERMISSION_MODE", "").strip()
        or DEFAULT_PERMISSION_MODE
    )


def _coerce_timeout(timeout: Any) -> float:
    if timeout is None:
        return _DEFAULT_TIMEOUT_SECONDS
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [
        getattr(timeout, attr, None)
        for attr in ("read", "write", "connect", "pool", "timeout")
    ]
    numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
    return max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS


# ── ACP tool-progress normalization ─────────────────────────────────
# Map Claude Code tool names (carried in update._meta.claudeCode.toolName)
# to the Hermes tool names that gateway/display key on (build_tool_preview,
# get_tool_emoji, the terminal fenced-block renderer). G1 trace confirmed
# the adapter (claude-agent-acp 0.48.0) uses these Claude names.
_ACP_TOOL_NAME_MAP = {
    "Bash": "terminal",
    "Read": "read_file",
    "Write": "write_file",
    "Edit": "patch",
    "MultiEdit": "patch",
    "Glob": "search_files",
    "Grep": "search_files",
    "shell": "terminal",
    "read": "read_file",
    "write": "write_file",
    "glob": "search_files",
    "grep": "search_files",
}
# Claude's rawInput arg keys differ from what build_tool_preview expects:
# Claude file tools use `file_path`, Hermes previews key on `path`. Without
# this remap the Read/Write/Edit progress preview renders blank (E-2/G1).
# Bash (`command`) and Grep (`pattern`) already align — left untouched.
_ACP_ARG_KEY_MAP = {
    "read_file": {"file_path": "path"},
    "write_file": {"file_path": "path"},
    "patch": {"file_path": "path"},
}


class ACPStreamUpdate:
    """Typed carrier for a raw ACP ``session/update`` attached to a heartbeat
    chunk (``choices=[]``). Lets the consumer render tool-progress WITHOUT
    faking an OpenAI ``delta.tool_calls`` (which would have the wrong lifecycle
    semantics and risk re-execution). Only ``tool_call``/``tool_call_update``
    carry one; ``plan``/``usage_update``/unknown stay plain heartbeats."""

    __slots__ = ("kind", "update")

    def __init__(self, kind: str, update: dict[str, Any]) -> None:
        self.kind = kind
        self.update = update


def _acp_tool_started(
    update: dict[str, Any], seen: set[str]
) -> tuple[str, dict[str, Any]] | None:
    """Normalize one ACP ``session/update`` into ``(hermes_tool_name, args)``
    for tool-progress, or ``None`` if it is not a render-worthy tool start.

    Observed event model (per ``toolCallId``): the bare ``tool_call``
    (pending) carries an EMPTY ``rawInput`` — the command/path/pattern arrive
    on the first ``tool_call_update``. So we emit from the first event that
    actually carries args, and dedup by ``toolCallId`` (``seen``) to fire once
    per tool. Output-bearing events (``rawOutput``/``toolResponse``, no
    ``rawInput``) are not starts and never leak their payload here.
    """
    if not isinstance(update, dict):
        return None
    kind = str(update.get("sessionUpdate") or "").strip()
    if kind not in ("tool_call", "tool_call_update"):
        return None
    raw_meta = update.get("_meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    raw_vendor_meta = meta.get("claudeCode")
    vendor_meta: dict[str, Any] = raw_vendor_meta if isinstance(raw_vendor_meta, dict) else {}
    raw_kiro_meta = meta.get("kiro")
    if not vendor_meta and isinstance(raw_kiro_meta, dict):
        vendor_meta = raw_kiro_meta
    raw_name = vendor_meta.get("toolName")
    if not isinstance(raw_name, str) or not raw_name:
        return None
    raw_args = update.get("rawInput")
    if not isinstance(raw_args, dict) or not raw_args:
        return None  # no args yet (pending) — wait for the update that carries them
    tool_call_id = update.get("toolCallId")
    if isinstance(tool_call_id, str) and tool_call_id:
        if tool_call_id in seen:
            return None  # already emitted a start for this tool
        seen.add(tool_call_id)
    mapped = _ACP_TOOL_NAME_MAP.get(raw_name, raw_name)
    key_map = _ACP_ARG_KEY_MAP.get(mapped)
    if key_map:
        args = {key_map.get(k, k): v for k, v in raw_args.items()}
    else:
        args = dict(raw_args)
    return mapped, args


def _emit_acp_tool_progress(chunk: Any, agent: Any, seen: set[str]) -> None:
    """Consumer-side: if ``chunk`` carries an ``ACPStreamUpdate`` for a tool
    start, fire ``agent.tool_progress_callback("tool.started", name, preview,
    args)`` once per tool. Never raises — a broken progress callback must not
    break the turn (mirrors agent/tool_executor.py:434-439). Output-bearing
    fields are never forwarded (only normalized ``args`` from ``rawInput``)."""
    acp_upd = getattr(chunk, "acp_update", None)
    if acp_upd is None:
        return
    cb = getattr(agent, "tool_progress_callback", None)
    if not cb:
        return
    try:
        update = getattr(acp_upd, "update", None)
        if not isinstance(update, dict):
            return
        started = _acp_tool_started(update, seen)
        if not started:
            return
        name, args = started
        from agent.display import build_tool_preview  # lazy: avoid import cycle
        preview = build_tool_preview(name, args)
        cb("tool.started", name, preview, args)
    except Exception as cb_err:  # noqa: BLE001
        logger.debug("ACP tool progress callback error: %s", cb_err)


class ACPSubprocessClientError(RuntimeError):
    """Structured ACP client error.

    ``reason`` is a coarse category consumed by the error-classifier bridge:
    one of ``startup``, ``auth``, ``unavailable``,
    ``rate_limit``, ``timeout``, ``protocol``, ``unknown``.
    """

    def __init__(self, message: str, *, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = reason


class _ACPChatCompletions:
    def __init__(self, client: "ACPSubprocessClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "ACPSubprocessClient"):
        self.completions = _ACPChatCompletions(client)


class ACPSubprocessClient:
    """Persistent-session, OpenAI-client-compatible facade for an ACP adapter."""

    supports_streaming = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        permission_mode: str | None = None,
        set_permission_mode: bool = True,
        provider_label: str = "claude-agent-acp",
        resume_session_id: str | None = None,
        timeout: Any = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key or "claude-agent-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = _resolve_command(acp_command or command)
        # Distinguish "not provided" (-> no extra args; the adapter bin needs
        # none) from an explicit list. claude-agent-acp takes no CLI args.
        if acp_args is not None:
            self._args = list(acp_args)
        elif args is not None:
            self._args = list(args)
        else:
            self._args = []
        self._cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self._permission_mode = _resolve_permission_mode(permission_mode)
        self._set_permission_mode = bool(set_permission_mode)
        self._provider_label = provider_label
        self._default_timeout = _coerce_timeout(timeout)

        self.chat = _ACPChatNamespace(self)
        self.is_closed = False

        # Persistent process / session state.
        self._proc: subprocess.Popen[str] | None = None
        self._proc_lock = threading.Lock()
        # One JSON-RPC request owns the shared inbox at a time. Streaming
        # prompts hold this lock for their full generator lifetime.
        self._request_lock = threading.RLock()
        # Serialize writes to the adapter's stdin: the worker thread sends
        # session/prompt while the outer poll loop may send session/cancel
        # cross-thread (see cancel()).
        self._send_lock = threading.Lock()
        self._inbox: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._stderr_tail: list[str] = []
        self._next_id = 0
        self._initialized = False

        # ACP session linkage. If resume_session_id is set, _ensure_session
        # loads that existing ACP session (session/load) instead of creating a
        # new one — so a caller can resume the same server-side session after
        # the client object is rebuilt (e.g. an in-session model switch).
        self.acp_session_id: str | None = None
        self._resume_session_id = (resume_session_id or "").strip() or None
        self.resumed = False
        # Set by _ensure_session when session/load of a prior ACP session
        # FAILED and we fell back to session/new: {"sid": <prior acp id>,
        # "reason": <cause>}. Surfaced on usage (acp_resume_failed) so a caller
        # can tell the user the live context was lost. None otherwise.
        self.resume_failed: dict[str, Any] | None = None
        self._delivered_count = 0          # how many OpenAI messages already sent
        self._last_delta_signature: str | None = None

        # Billing: the adapter reports cumulative session cost in USD.
        self._cumulative_cost_usd = 0.0
        self._context_used = 0
        self._context_size = 0
        self._vendor_credits: float | None = None
        self._context_usage_percentage: float | None = None
        self._active_model_id: str | None = None

    # ------------------------------------------------------------------ #
    # Process lifecycle
    # ------------------------------------------------------------------ #
    def _spawn(self) -> subprocess.Popen[str]:
        try:
            proc = subprocess.Popen(
                [self._command] + self._args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._cwd,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise ACPSubprocessClientError(
                f"Could not start ACP command '{self._command}'. Install the adapter "
                "(npm i -g @agentclientprotocol/claude-agent-acp) or set "
                "HERMES_CLAUDE_AGENT_ACP_COMMAND.",
                reason="startup",
            ) from exc
        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise ACPSubprocessClientError(
                "ACP adapter did not expose stdin/stdout pipes.", reason="startup"
            )

        def _stdout_reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._inbox.put(json.loads(line))
                except Exception:
                    self._inbox.put({"__raw__": line})

        def _stderr_reader() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                tail = self._stderr_tail
                tail.append(line.rstrip("\n"))
                if len(tail) > 80:
                    del tail[: len(tail) - 80]

        threading.Thread(target=_stdout_reader, daemon=True).start()
        threading.Thread(target=_stderr_reader, daemon=True).start()
        return proc

    def close(self) -> None:
        with self._proc_lock:
            proc = self._proc
            self._proc = None
        self.is_closed = True
        self._initialized = False
        self.acp_session_id = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def cancel(self) -> None:
        """Cleanly interrupt the in-flight turn via the ACP ``session/cancel``
        notification. The adapter aborts tool calls / LLM requests, flushes
        pending updates, then completes ``session/prompt`` with
        ``stopReason=cancelled``. Best-effort; never raises. Preferred over
        ``close()`` because it keeps the subprocess + ACP session alive so the
        next turn resumes instead of forcing a fresh ``session/new``."""
        try:
            if self._proc is not None and self._proc.poll() is None and self.acp_session_id:
                self._notify("session/cancel", {"sessionId": self.acp_session_id})
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # JSON-RPC plumbing
    # ------------------------------------------------------------------ #
    def _send(self, obj: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        with self._send_lock:
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float,
        text_parts: list[str] | None = None,
        reasoning_parts: list[str] | None = None,
    ) -> Any:
        with self._request_lock:
            return self._request_unlocked(
                method, params, timeout_seconds=timeout_seconds,
                text_parts=text_parts, reasoning_parts=reasoning_parts,
            )

    def _request_unlocked(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float,
        text_parts: list[str] | None = None,
        reasoning_parts: list[str] | None = None,
    ) -> Any:
        assert self._proc is not None
        self._next_id += 1
        req_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                break
            try:
                msg = self._inbox.get(timeout=0.1)
            except queue.Empty:
                continue

            if isinstance(msg, dict) and msg.get("id") == req_id and (
                "result" in msg or "error" in msg
            ):
                if "error" in msg:
                    err = msg.get("error") or {}
                    raise self._classify_rpc_error(method, err)
                return msg.get("result")

            self._handle_inbound(msg, text_parts, reasoning_parts)

        # Timed out or process exited.
        tail = "\n".join(self._stderr_tail).strip()
        if self._proc.poll() is not None:
            raise ACPSubprocessClientError(
                f"ACP adapter exited before responding to {method}. stderr:\n{tail}",
                reason=self._classify_stderr(tail),
            )
        raise ACPSubprocessClientError(
            f"Timed out waiting for ACP response to {method}.", reason="timeout"
        )

    def _handle_inbound(
        self,
        msg: dict[str, Any],
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
    ) -> None:
        if not isinstance(msg, dict):
            return
        method = msg.get("method")
        if not isinstance(method, str):
            return  # a result for some other id, or raw line — ignore here

        params = msg.get("params") or {}
        msg_id = msg.get("id")

        if method == "session/update" and msg_id is None:
            self._consume_session_update(params, text_parts, reasoning_parts)
            return

        if method == "_kiro.dev/metadata" and msg_id is None:
            self._consume_kiro_metadata(params)
            return

        if msg_id is None:
            return  # other notifications — ignore

        # Inbound request that needs a response.
        if method == "session/request_permission":
            # With bypassPermissions the adapter should not ask; if it does
            # (e.g. operator set a stricter mode) honour the chosen posture by
            # allowing. A future stricter mode would route this to Hermes safety.
            self._send(self._permission_response(msg_id, params))
            return
        if method in ("fs/read_text_file", "fs/write_text_file"):
            # claude-agent-acp performs its own fs ops; if a client-fs request
            # ever arrives, refuse rather than silently granting.
            self._send({
                "jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": "client fs not provided"},
            })
            return
        # Unknown inbound request.
        self._send({
            "jsonrpc": "2.0", "id": msg_id,
            "error": {"code": -32601, "message": f"method '{method}' not handled"},
        })

    def _permission_response(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        options = params.get("options") if isinstance(params, dict) else None
        chosen = None
        if isinstance(options, list):
            for opt in options:
                if isinstance(opt, dict) and "allow" in str(opt.get("kind", "")).lower():
                    chosen = opt.get("optionId")
                    break
            if chosen is None and options and isinstance(options[0], dict):
                chosen = options[0].get("optionId")
        if chosen is not None:
            return {"jsonrpc": "2.0", "id": msg_id,
                    "result": {"outcome": {"outcome": "selected", "optionId": chosen}}}
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": {"outcome": {"outcome": "cancelled"}}}

    def _consume_session_update(
        self,
        params: dict[str, Any],
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
    ) -> None:
        update = (params or {}).get("update") or {}
        kind = str(update.get("sessionUpdate") or "").strip()
        if kind == "agent_message_chunk" and text_parts is not None:
            content = update.get("content") or {}
            if isinstance(content, dict) and content.get("type") == "text":
                text_parts.append(str(content.get("text") or ""))
        elif kind == "agent_thought_chunk" and reasoning_parts is not None:
            content = update.get("content") or {}
            if isinstance(content, dict) and content.get("type") == "text":
                reasoning_parts.append(str(content.get("text") or ""))
        elif kind == "usage_update":
            used = update.get("used")
            size = update.get("size")
            if isinstance(used, (int, float)):
                self._context_used = int(used)
            if isinstance(size, (int, float)):
                self._context_size = int(size)
            cost = update.get("cost") or {}
            amount = cost.get("amount") if isinstance(cost, dict) else None
            if isinstance(amount, (int, float)):
                self._cumulative_cost_usd = float(amount)

    def _consume_kiro_metadata(self, params: dict[str, Any]) -> None:
        """Capture Kiro-native metering without mislabelling credits as USD."""
        metadata = params.get("metadata") if isinstance(params.get("metadata"), dict) else params
        credits = metadata.get("meteringUsage") if isinstance(metadata, dict) else None
        context = metadata.get("contextUsagePercentage") if isinstance(metadata, dict) else None
        if isinstance(credits, (int, float)):
            self._vendor_credits = float(credits)
        if isinstance(context, (int, float)):
            self._context_usage_percentage = float(context)

    # ------------------------------------------------------------------ #
    # Error classification helpers (feeds stage-8 bridge)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _classify_stderr(stderr_text: str) -> str:
        low = stderr_text.lower()
        if any(k in low for k in ("unauthorized", "not logged in", "login", "authenticate", "401")):
            return "auth"
        if any(k in low for k in ("rate limit", "429", "quota", "credit", "usage limit")):
            return "rate_limit"
        if any(k in low for k in ("econnrefused", "enotfound", "etimedout", "503", "502", "unavailable")):
            return "unavailable"
        return "startup"

    def _classify_rpc_error(self, method: str, err: dict[str, Any]) -> ACPSubprocessClientError:
        message = str(err.get("message") or err)
        low = message.lower()
        if any(k in low for k in ("unauthorized", "not logged in", "authenticate", "login", "401")):
            reason = "auth"
        elif any(k in low for k in ("rate limit", "429", "quota", "credit", "usage limit")):
            reason = "rate_limit"
        elif any(k in low for k in ("unavailable", "503", "502", "overloaded")):
            reason = "unavailable"
        else:
            reason = "protocol"
        return ACPSubprocessClientError(
            f"ACP {method} failed: {message}", reason=reason
        )

    # ------------------------------------------------------------------ #
    # Session setup
    # ------------------------------------------------------------------ #
    def _ensure_session(self, timeout_seconds: float) -> None:
        with self._proc_lock:
            if self._proc is not None and self._proc.poll() is None and self._initialized:
                return
            # Messages from a dead process must never satisfy requests sent to
            # its successor.
            if self._proc is not None:
                while True:
                    try:
                        self._inbox.get_nowait()
                    except queue.Empty:
                        break
                self._stderr_tail.clear()
            self._proc = self._spawn()
        self.is_closed = False

        init_result = self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}},
                "clientInfo": {"name": "hermes-agent", "title": "Hermes Agent", "version": "0.0.0"},
            },
            timeout_seconds=timeout_seconds,
        )
        # Observability: surface the loadSession capability flag from the raw
        # initialize result. Log-only — capability gating is out of scope; the
        # fallback path (session/load failure -> session/new) is the backstop
        # regardless of what the adapter advertises here. Local var only:
        # nothing else consumes this state.
        _init_result = init_result if isinstance(init_result, dict) else {}
        _agent_caps = _init_result.get("agentCapabilities") or _init_result.get("capabilities") or {}
        logger.info(
            "ACP initialize complete: agentCapabilities.loadSession=%s",
            _agent_caps.get("loadSession") if isinstance(_agent_caps, dict) else None,
        )
        session_id = ""
        self.resumed = False
        # Recomputed on every fresh (re)initialization: a successful load leaves
        # this None; a failed load sets it below. The early-return path (already
        # initialized) keeps whatever this init decided.
        self.resume_failed = None
        _session_new_cause = "no_resume_sid"
        # Try to resume an existing ACP session first (survives a client
        # rebuild / model switch). loadSession replays history server-side; we drop those
        # replayed notifications (text_parts is None during setup).
        if self._resume_session_id:
            logger.info(
                "Attempting ACP session/load for sessionId=%s (resuming prior context).",
                self._resume_session_id,
            )
            try:
                self._request(
                    "session/load",
                    {"sessionId": self._resume_session_id, "cwd": self._cwd, "mcpServers": []},
                    timeout_seconds=timeout_seconds,
                )
                session_id = self._resume_session_id
                self.resumed = True
                logger.info(
                    "ACP session resumed via session/load (sessionId=%s) — "
                    "live context preserved, no seed bridge.",
                    self._resume_session_id,
                )
            except ACPSubprocessClientError as exc:
                # Session gone / load unsupported — fall back to a fresh session
                # rather than failing the turn. Context continuity is lost, but
                # this is explicit (resumed stays False), not silent. Log a
                # WARNING so a failed resume is visible rather than looking like
                # a silently-dropped session.
                logger.warning(
                    "ACP session/load FAILED for sessionId=%s (%s); ACP "
                    "session/new cause=load_failed — falling back to session/new. "
                    "The prior server-side context is not replayed.",
                    self._resume_session_id, exc,
                )
                session_id = ""
                _session_new_cause = "load_failed"
                # Record the failure so it can be surfaced on usage
                # (acp_resume_failed) — a caller can then tell the user the live
                # context was lost. The coarse reason classifier is enough for
                # user-facing text; the full exc already went to the WARNING.
                self.resume_failed = {
                    "sid": self._resume_session_id,
                    "reason": getattr(exc, "reason", None) or "load_failed",
                }

        if not session_id:
            if _session_new_cause == "no_resume_sid":
                logger.info(
                    "ACP session/new cause=no_resume_sid — no prior ACP session id "
                    "resolved for this client; starting a fresh session.",
                )
            session = self._request(
                "session/new",
                {"cwd": self._cwd, "mcpServers": []},
                timeout_seconds=timeout_seconds,
            ) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise ACPSubprocessClientError(
                    "session/new did not return a sessionId.", reason="protocol"
                )

        self.acp_session_id = session_id
        # Keep the respawn target current. This client instance is reused
        # turn-to-turn — _ensure_session runs again (and re-spawns the adapter
        # process) whenever the process died since the last turn
        # (`self._proc.poll() is not None`), NOT just at construction. Pointing
        # `_resume_session_id` at the session this instance actually established
        # means a mid-life respawn resumes what this instance is holding rather
        # than a stale id resolved once at construction.
        self._resume_session_id = session_id
        if self._set_permission_mode:
            self._apply_permission_mode(session_id, timeout_seconds)
        self._initialized = True
        # Fresh process: nothing delivered to this process yet.
        self._delivered_count = 0

    def _apply_permission_mode(self, session_id: str, timeout_seconds: float) -> None:
        """Set the adapter's permission posture deterministically (best-effort)."""
        try:
            self._request(
                "session/set_mode",
                {"sessionId": session_id, "modeId": self._permission_mode},
                timeout_seconds=min(timeout_seconds, 30.0),
            )
        except ACPSubprocessClientError:
            # Older adapters / unsupported mode: fall back to adapter default.
            # The cwd sandbox still bounds writes.
            pass

    def set_session_model(self, model_id: str, timeout_seconds: float | None = None) -> None:
        """Switch the model inside the live ACP session."""
        if not self.acp_session_id:
            return
        self._request(
            "session/set_model",
            {"sessionId": self.acp_session_id, "modelId": model_id},
            timeout_seconds=timeout_seconds or self._default_timeout,
        )

    # ------------------------------------------------------------------ #
    # Delta computation
    # ------------------------------------------------------------------ #
    def _compute_delta(self, messages: list[dict[str, Any]]) -> tuple[str, int]:
        """Return (prompt_text, new_delivered_count).

        Sends only the trailing un-responded turn: messages after the last
        assistant message. This is robust to Hermes-side compaction (which
        rewrites the prefix) because the stateful ACP session already holds the
        real context — we never replay the historical prefix.

        That holds only for a session that has the prefix: one this client
        already prompted, or one restored with session/load. A session this
        client just opened with session/new has nothing, yet the transcript may
        already hold earlier turns — Hermes builds such clients mid-conversation:
        the request client is closed after every stream error (a stale-stream
        kill included) and replaced, and the background review builds its own.
        Sending only the trailing turn there gave the model a bare continuation
        nudge with no task ("I don't have any prior task content in this
        session"). So such a session, like one whose session/load failed, gets
        the whole role-labelled transcript once; later turns are deltas again.
        """
        msgs = [m for m in (messages or []) if isinstance(m, dict)]
        last_assistant = -1
        for i, m in enumerate(msgs):
            if str(m.get("role") or "").lower() == "assistant":
                last_assistant = i
        session_lacks_prefix = not self.resumed and self._delivered_count == 0 and last_assistant >= 0
        replay_full_history = self.resume_failed is not None or session_lacks_prefix
        if replay_full_history:
            last_assistant = -1
        delta = msgs[last_assistant + 1:]
        # First turn (no prior assistant): include leading system messages so
        # Claude adopts Hermes' task framing.
        rendered: list[str] = []
        for m in delta:
            role = str(m.get("role") or "").lower()
            text = _render_message_content(m.get("content"))
            if not text:
                continue
            if replay_full_history:
                rendered.append(f"[{role.title() or 'Message'}]\n{text}")
            elif role == "system":
                rendered.append(f"[System instructions]\n{text}")
            elif role == "tool":
                rendered.append(f"[Tool result]\n{text}")
            else:
                rendered.append(text)
        return "\n\n".join(rendered).strip(), len(msgs)

    # ------------------------------------------------------------------ #
    # OpenAI-shim entry point
    # ------------------------------------------------------------------ #
    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: Any = None,
        tools: list[dict[str, Any]] | None = None,   # ignored: adapter uses own tools
        tool_choice: Any = None,                      # ignored
        stream: bool = False,
        **_: Any,
    ) -> Any:
        timeout_seconds = _coerce_timeout(timeout) if timeout is not None else self._default_timeout
        self._ensure_session(timeout_seconds)

        if model and model != self._provider_label and model != self._active_model_id:
            self.set_session_model(model, timeout_seconds=min(timeout_seconds, 30.0))
            self._active_model_id = model

        # The stateful ACP session already holds the full server-side
        # conversation, so only the new turn (the delta over what was already
        # delivered) is sent — see _compute_delta.
        prompt_text, new_count = self._compute_delta(messages or [])
        if not prompt_text:
            # Nothing new to send (e.g. a duplicate call). Send a minimal
            # continuation rather than re-prompting the model with old turns.
            prompt_text = "(continue)"

        if stream:
            # Returns a generator of OpenAI-shaped chunks. The stream is bounded
            # by genuine idle / adapter death, not a fixed wall-clock timeout, so
            # a long agentic turn that keeps emitting events never false-times-out.
            return self._prompt_stream(
                prompt_text, new_count, model, timeout_seconds,
            )

        cost_before = self._cumulative_cost_usd
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        self._request(
            "session/prompt",
            {
                "sessionId": self.acp_session_id,
                "prompt": [{"type": "text", "text": prompt_text}],
            },
            timeout_seconds=timeout_seconds,
            text_parts=text_parts,
            reasoning_parts=reasoning_parts,
        )
        self._delivered_count = new_count

        response_text = "".join(text_parts)
        reasoning_text = "".join(reasoning_parts)
        usage = self._build_usage(cost_before)
        assistant_message = SimpleNamespace(
            content=response_text,
            tool_calls=[],
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or self._provider_label,
        )

    def _build_usage(self, cost_before: float) -> Any:
        """Assemble the OpenAI-shaped usage namespace (shared by the blocking
        and streaming paths). ``acp_session_id`` is captured here, while the
        client is alive — reading it after the turn is racy (close() resets it
        to None)."""
        turn_cost = max(0.0, self._cumulative_cost_usd - cost_before)
        return SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=int(self._context_used or 0),
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            # ACP-native billing: authoritative USD reported by the adapter.
            cost_usd=turn_cost,
            total_cost_usd=self._cumulative_cost_usd,
            context_used=self._context_used,
            context_size=self._context_size,
            vendor_credits=self._vendor_credits,
            context_usage_percentage=self._context_usage_percentage,
            acp_session_id=self.acp_session_id,
            # Resume flags, shared by the blocking and streaming paths (both call
            # _build_usage) so usage carries identical resume signals regardless
            # of stream=True/False:
            #   acp_resumed      — prior ACP session was replayed via session/load
            #   acp_resume_failed — {"sid","reason"} when session/load FAILED and
            #                       we fell back to session/new (live context
            #                       lost), else None. A caller can warn the user.
            acp_resumed=bool(self.resumed),
            acp_resume_failed=self.resume_failed,
        )

    # ------------------------------------------------------------------ #
    # Streaming — yield OpenAI-shaped chunks per session/update
    # ------------------------------------------------------------------ #
    def _prompt_stream(self, prompt_text, new_count, model, timeout_seconds):
        """Generator: stream one turn as OpenAI ``ChatCompletionChunk``-shaped
        objects.

        Yields a content/reasoning delta chunk per ``agent_message_chunk`` /
        ``agent_thought_chunk`` notification, a heartbeat chunk (empty
        ``choices``) for ``tool_call`` / ``tool_call_update`` / ``plan`` /
        ``usage_update`` (so the consumer's idle watchdog resets while the agent
        works silently with tools), and one final chunk carrying the mapped
        ``finish_reason`` and the usage namespace.

        Bounded by genuine idle (no notification for ``timeout_seconds``) or
        adapter death — NOT total wall-clock — so a long agentic turn that keeps
        emitting events never false-times-out."""
        with self._request_lock:
            cost_before = self._cumulative_cost_usd
            self._next_id += 1
            req_id = self._next_id
            self._send({
                "jsonrpc": "2.0", "id": req_id, "method": "session/prompt",
                "params": {
                    "sessionId": self.acp_session_id,
                    "prompt": [{"type": "text", "text": prompt_text}],
                },
            })
            stop_reason = ""
            idle_deadline = time.monotonic() + timeout_seconds
            while True:
                if self._proc is None or self._proc.poll() is not None:
                    tail = "\n".join(self._stderr_tail).strip()
                    raise ACPSubprocessClientError(
                        f"ACP adapter exited mid-stream. stderr:\n{tail}",
                        reason=self._classify_stderr(tail),
                    )
                if time.monotonic() > idle_deadline:
                    raise ACPSubprocessClientError(
                        f"ACP stream idle for {timeout_seconds:.0f}s with no events.",
                        reason="timeout",
                    )
                try:
                    msg = self._inbox.get(timeout=0.1)
                except queue.Empty:
                    continue
                idle_deadline = time.monotonic() + timeout_seconds
                if isinstance(msg, dict) and msg.get("id") == req_id and (
                    "result" in msg or "error" in msg
                ):
                    if "error" in msg:
                        raise self._classify_rpc_error("session/prompt", msg.get("error") or {})
                    result = msg.get("result") or {}
                    stop_reason = str(result.get("stopReason") or "").strip()
                    break
                chunk = self._stream_chunk_from_inbound(msg, model)
                if chunk is None:
                    continue
                yield chunk

            self._delivered_count = new_count
            finish_reason = _ACP_STOP_REASON_MAP.get(stop_reason.lower(), "stop")
            yield self._final_chunk(model, finish_reason, self._build_usage(cost_before))

    def _stream_chunk_from_inbound(self, msg: dict[str, Any], model: str | None) -> Any:
        if not isinstance(msg, dict):
            return None
        method = msg.get("method")
        if not isinstance(method, str):
            return None  # a result for some other id / raw line
        msg_id = msg.get("id")
        if method == "session/update" and msg_id is None:
            return self._stream_chunk_from_update(msg.get("params") or {}, model)
        if msg_id is None:
            return None  # other notification — ignore
        # Inbound request needing a response (permission / fs) — reuse the
        # blocking handler, then emit a heartbeat so the watchdog stays fresh.
        self._handle_inbound(msg, None, None)
        return self._heartbeat_chunk(model)

    def _stream_chunk_from_update(self, params: dict[str, Any], model: str | None) -> Any:
        update = (params or {}).get("update") or {}
        kind = str(update.get("sessionUpdate") or "").strip()
        if kind == "agent_message_chunk":
            content = update.get("content") or {}
            if isinstance(content, dict) and content.get("type") == "text":
                return self._content_chunk(model, str(content.get("text") or ""))
        elif kind == "agent_thought_chunk":
            content = update.get("content") or {}
            if isinstance(content, dict) and content.get("type") == "text":
                return self._reasoning_chunk(model, str(content.get("text") or ""))
        elif kind == "usage_update":
            # Keep cost/context bookkeeping in sync (reuse the blocking logic).
            self._consume_session_update(params, None, None)
        elif kind in ("tool_call", "tool_call_update"):
            # Still a heartbeat (no visible delta, idle-watchdog resets), but
            # carry the raw update so the consumer can render tool-progress.
            hb = self._heartbeat_chunk(model)
            hb.acp_update = ACPStreamUpdate(kind, update)
            return hb
        # plan / unknown → plain heartbeat: a yielded event so the consumer's
        # idle watchdog resets, but no visible delta and no tool-progress.
        return self._heartbeat_chunk(model)

    def _content_chunk(self, model: str | None, text: str) -> Any:
        return SimpleNamespace(
            model=model or self._provider_label,
            choices=[SimpleNamespace(
                index=0, finish_reason=None,
                delta=SimpleNamespace(role="assistant", content=text,
                                      reasoning_content=None, tool_calls=None),
            )],
            usage=None,
        )

    def _reasoning_chunk(self, model: str | None, text: str) -> Any:
        return SimpleNamespace(
            model=model or self._provider_label,
            choices=[SimpleNamespace(
                index=0, finish_reason=None,
                delta=SimpleNamespace(role="assistant", content=None,
                                      reasoning_content=text, tool_calls=None),
            )],
            usage=None,
        )

    def _heartbeat_chunk(self, model: str | None) -> Any:
        return SimpleNamespace(model=model or self._provider_label, choices=[], usage=None)

    def _final_chunk(self, model: str | None, finish_reason: str, usage: Any) -> Any:
        return SimpleNamespace(
            model=model or self._provider_label,
            choices=[SimpleNamespace(
                index=0, finish_reason=finish_reason,
                delta=SimpleNamespace(role="assistant", content=None,
                                      reasoning_content=None, tool_calls=None),
            )],
            usage=usage,
        )

