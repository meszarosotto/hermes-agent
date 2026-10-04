"""Unit tests for ACPSubprocessClient logic (no live adapter required).

Covers the delta invariant, arg/command resolution, permission-mode
resolution, error classification, streaming chunk shaping, tool-progress
normalization, session resume, and the usage/cost contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.acp_subprocess_client import (
    ACPSubprocessClient,
    ACPSubprocessClientError,
    ACPStreamUpdate,
    _acp_tool_started,
    _emit_acp_tool_progress,
    _resolve_command,
    _resolve_permission_mode,
    _coerce_timeout,
)


def _client(**kw):
    return ACPSubprocessClient(acp_cwd="/tmp", **kw)


# ── arg / command / mode resolution ──────────────────────────────────

def test_default_command():
    with patch.dict("os.environ", {}, clear=False):
        c = _client()
    assert c._command == "claude-agent-acp"


def test_command_env_override():
    with patch.dict("os.environ", {"HERMES_CLAUDE_AGENT_ACP_COMMAND": "/opt/acp"}, clear=False):
        c = _client()
    assert c._command == "/opt/acp"


def test_args_default_empty():
    c = _client()
    assert c._args == []  # adapter bin takes no args


def test_args_none_vs_explicit():
    assert _client(acp_args=None)._args == []
    assert _client(acp_args=[])._args == []
    assert _client(acp_args=["--x"])._args == ["--x"]
    assert _client(args=["--legacy"])._args == ["--legacy"]
    assert _client(acp_args=["--new"], args=["--old"])._args == ["--new"]


def test_default_permission_mode_is_bypass():
    with patch.dict("os.environ", {}, clear=False):
        assert _resolve_permission_mode(None) == "bypassPermissions"


def test_permission_mode_explicit_and_env():
    assert _resolve_permission_mode("plan") == "plan"
    with patch.dict("os.environ", {"HERMES_CLAUDE_AGENT_ACP_PERMISSION_MODE": "default"}, clear=False):
        assert _resolve_permission_mode(None) == "default"


def test_coerce_timeout():
    assert _coerce_timeout(None) == 900.0
    assert _coerce_timeout(12.5) == 12.5

    class _T:
        read = 30.0
        write = 5.0
        connect = None
    assert _coerce_timeout(_T()) == 30.0


# ── delta invariant ──────────────────────────────────────────────────

def test_delta_first_turn_includes_system_and_user():
    c = _client()
    msgs = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Hello"},
    ]
    text, count = c._compute_delta(msgs)
    assert "Be terse." in text
    assert "Hello" in text
    assert count == 2


def test_delta_second_turn_sends_only_trailing_user():
    c = _client()
    c._delivered_count = 2  # this session already received the first exchange
    msgs = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    text, count = c._compute_delta(msgs)
    # Only the trailing un-responded user turn — NOT the historical prefix.
    assert text == "second question"
    assert "first question" not in text
    assert "first answer" not in text
    assert count == 4


def test_delta_after_compaction_only_sends_new_user_turn():
    # Hermes compaction rewrites the prefix into a summary; the stateful ACP
    # session already holds real context, so we must send only the new turn.
    c = _client()
    c._delivered_count = 7  # prompted before compaction shortened the transcript
    msgs = [
        {"role": "system", "content": "[summary of earlier conversation]"},
        {"role": "assistant", "content": "older answer kept by compaction"},
        {"role": "user", "content": "brand new question"},
    ]
    text, count = c._compute_delta(msgs)
    assert text == "brand new question"
    assert "summary" not in text
    assert count == 3


def test_delta_tool_results_labelled():
    c = _client()
    c._delivered_count = 1
    msgs = [
        {"role": "assistant", "content": "prior"},
        {"role": "tool", "content": "tool output here"},
        {"role": "user", "content": "ok continue"},
    ]
    text, _ = c._compute_delta(msgs)
    assert "[Tool result]" in text
    assert "tool output here" in text
    assert "ok continue" in text


def test_resume_failure_reseeds_full_role_labelled_history():
    c = _client()
    c.resume_failed = {"sid": "lost", "reason": "protocol"}
    text, count = c._compute_delta([
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ])
    assert "[System]\nBe terse." in text
    assert "[User]\nfirst question" in text
    assert "[Assistant]\nfirst answer" in text
    assert "[User]\nsecond question" in text
    assert count == 4


# ── error classification (feeds stage-8 bridge) ──────────────────────

def test_classify_stderr_auth():
    assert ACPSubprocessClient._classify_stderr("Error: Unauthorized (401)") == "auth"
    assert ACPSubprocessClient._classify_stderr("please run /login") == "auth"


def test_classify_stderr_rate_limit():
    assert ACPSubprocessClient._classify_stderr("429 rate limit exceeded") == "rate_limit"
    assert ACPSubprocessClient._classify_stderr("monthly credit exhausted") == "rate_limit"


def test_classify_stderr_unavailable():
    assert ACPSubprocessClient._classify_stderr("connect ECONNREFUSED") == "unavailable"
    assert ACPSubprocessClient._classify_stderr("503 service unavailable") == "unavailable"


def test_classify_stderr_startup_default():
    assert ACPSubprocessClient._classify_stderr("some random crash") == "startup"


def test_classify_rpc_error_reasons():
    c = _client()
    assert c._classify_rpc_error("session/prompt", {"message": "Unauthorized"}).reason == "auth"
    assert c._classify_rpc_error("session/prompt", {"message": "rate limit"}).reason == "rate_limit"
    assert c._classify_rpc_error("session/prompt", {"message": "service unavailable"}).reason == "unavailable"
    assert c._classify_rpc_error("session/prompt", {"message": "weird"}).reason == "protocol"


def test_startup_error_when_command_missing():
    c = _client(acp_command="/nonexistent/acp-binary-xyz")
    with pytest.raises(ACPSubprocessClientError) as ei:
        c._spawn()
    assert ei.value.reason == "startup"


# ── usage / cost capture (billing contract) ──────────────────────────

def test_usage_update_captures_cost_and_context():
    c = _client()
    c._consume_session_update(
        {"update": {"sessionUpdate": "usage_update", "used": 27624, "size": 1000000,
                    "cost": {"amount": 0.0791, "currency": "USD"}}},
        None, None,
    )
    assert c._cumulative_cost_usd == pytest.approx(0.0791)
    assert c._context_used == 27624
    assert c._context_size == 1000000


def test_kiro_metadata_keeps_credits_separate_from_usd():
    c = _client()
    c._consume_kiro_metadata({"meteringUsage": 2.75, "contextUsagePercentage": 41.5})
    usage = c._build_usage(0.0)
    assert usage.vendor_credits == pytest.approx(2.75)
    assert usage.context_usage_percentage == pytest.approx(41.5)
    assert usage.cost_usd == 0.0


def test_kiro_tool_name_is_normalized_for_hermes_progress():
    started = _acp_tool_started({
        "sessionUpdate": "tool_call_update",
        "toolCallId": "k1",
        "_meta": {"kiro": {"toolName": "shell"}},
        "rawInput": {"command": "pwd"},
    }, set())
    assert started == ("terminal", {"command": "pwd"})


def test_kiro_tool_progress_callback_is_emitted_once():
    events = []
    agent = SimpleNamespace(
        tool_progress_callback=lambda *args: events.append(args),
    )
    update = {
        "sessionUpdate": "tool_call_update",
        "toolCallId": "k1",
        "_meta": {"kiro": {"toolName": "shell"}},
        "rawInput": {"command": "pwd"},
    }
    chunk = SimpleNamespace(acp_update=ACPStreamUpdate("tool_call_update", update))
    seen = set()
    _emit_acp_tool_progress(chunk, agent, seen)
    _emit_acp_tool_progress(chunk, agent, seen)
    assert len(events) == 1
    assert events[0][0:2] == ("tool.started", "terminal")


def test_agent_message_chunk_accumulates_text():
    c = _client()
    parts: list[str] = []
    c._consume_session_update(
        {"update": {"sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "hello "}}},
        parts, None,
    )
    c._consume_session_update(
        {"update": {"sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "world"}}},
        parts, None,
    )
    assert "".join(parts) == "hello world"


def test_resume_session_id_stored():
    assert _client(resume_session_id="abc-123")._resume_session_id == "abc-123"
    assert _client(resume_session_id="  ")._resume_session_id is None
    assert _client()._resume_session_id is None


# ── respawn honesty: resume target tracks the live session ────────────
#
# _ensure_session runs on EVERY turn (called from _create_chat_completion),
# not just once at client construction. Its early-return guard —
# ``self._proc is not None and self._proc.poll() is None and
# self._initialized`` — reuses the live process/session when both are still
# alive, but falls through to a full respawn (new subprocess + init) the
# moment the adapter process has died between turns. Before this fix,
# ``_resume_session_id`` was set once in ``__init__`` and never updated, so a
# mid-life respawn always retried whatever the factory resolved at
# construction time — NOT the session this client instance actually
# established — silently losing context with no honest signal in two ways:
# (a) a genuinely fresh session (started with resume_session_id=None) reborn
# as a second fresh session with no resume_failed flag at all (cause=
# no_resume_sid never even attempts a load); (b) a second, distinct loss
# collapsing onto the same stale sid as an earlier load_failed, which
# conversation_loop's dedup (keyed by sid) would then silently swallow.

def _fake_proc(alive: bool = True) -> MagicMock:
    proc = MagicMock()
    proc.poll.return_value = None if alive else 1
    return proc


def test_ensure_session_respawn_resumes_current_session_not_stale_id():
    """Fresh session (no resume_session_id at construction): after the first
    _ensure_session establishes a session via session/new, a respawn (process
    death) must attempt session/load against THAT session — not silently
    session/new again with the original (None) resume target."""
    c = _client()
    assert c._resume_session_id is None
    proc = _fake_proc(alive=True)

    calls: list[tuple[str, dict]] = []

    def _request_gen1(method, params, *, timeout_seconds, text_parts=None, reasoning_parts=None):
        calls.append((method, dict(params)))
        if method == "session/new":
            return {"sessionId": "SESS-GEN-1"}
        return {}

    with patch.object(c, "_spawn", return_value=proc), \
         patch.object(c, "_request", side_effect=_request_gen1):
        c._ensure_session(5.0)

    assert c.acp_session_id == "SESS-GEN-1"
    assert c.resume_failed is None
    # The fix under test: the client now tracks its OWN live session as the
    # future resume target, not the (None) value resolved at construction.
    assert c._resume_session_id == "SESS-GEN-1"

    # Simulate the adapter process dying between turns.
    proc.poll.return_value = 1
    calls.clear()

    def _request_gen2(method, params, *, timeout_seconds, text_parts=None, reasoning_parts=None):
        calls.append((method, dict(params)))
        if method == "session/load":
            return {}
        return {}

    with patch.object(c, "_spawn", return_value=proc), \
         patch.object(c, "_request", side_effect=_request_gen2):
        c._ensure_session(5.0)

    load_calls = [p for m, p in calls if m == "session/load"]
    assert load_calls == [{"sessionId": "SESS-GEN-1", "cwd": c._cwd, "mcpServers": []}]
    assert c.resumed is True
    assert c.resume_failed is None


def test_ensure_session_respawn_after_load_failure_retries_current_not_stale():
    """Stale resume id at construction fails to load, so a fresh session gets
    created (resume_failed records the STALE sid). A later respawn (process
    death) must retry loading the FRESH session this instance now holds — not
    the original stale id again — so a second, distinct loss surfaces under
    its own (new) sid instead of being swallowed by sid-based notice dedup."""
    c = _client(resume_session_id="STALE-X")
    proc = _fake_proc(alive=True)

    def _request_gen1(method, params, *, timeout_seconds, text_parts=None, reasoning_parts=None):
        if method == "session/load":
            raise ACPSubprocessClientError("not found", reason="protocol")
        if method == "session/new":
            return {"sessionId": "FRESH-Y"}
        return {}

    with patch.object(c, "_spawn", return_value=proc), \
         patch.object(c, "_request", side_effect=_request_gen1):
        c._ensure_session(5.0)

    assert c.acp_session_id == "FRESH-Y"
    assert c.resume_failed == {"sid": "STALE-X", "reason": "protocol"}
    assert c._resume_session_id == "FRESH-Y"  # tracks the live session, not X

    # Process dies; the next turn's _ensure_session respawns.
    proc.poll.return_value = 1
    seen_load_sids: list[str] = []

    def _request_gen2(method, params, *, timeout_seconds, text_parts=None, reasoning_parts=None):
        if method == "session/load":
            seen_load_sids.append(params.get("sessionId"))
            raise ACPSubprocessClientError("not found again", reason="protocol")
        if method == "session/new":
            return {"sessionId": "FRESH-Z"}
        return {}

    with patch.object(c, "_spawn", return_value=proc), \
         patch.object(c, "_request", side_effect=_request_gen2):
        c._ensure_session(5.0)

    assert seen_load_sids == ["FRESH-Y"]  # retried the CURRENT session, not STALE-X
    assert c.resume_failed == {"sid": "FRESH-Y", "reason": "protocol"}
    assert c.acp_session_id == "FRESH-Z"


def test_ensure_session_resets_resume_failed_on_successful_retry():
    """No-leak invariant: resume_failed is recomputed on every fresh
    (re)initialization, so a failed load followed
    by a later successful load/resume must NOT leave a stale resume_failed
    flag set — a fresh _ensure_session that succeeds always resets it to
    None at the top before deciding the outcome of THIS attempt."""
    c = _client(resume_session_id="X")
    proc = _fake_proc(alive=True)

    def _request_fail(method, params, *, timeout_seconds, text_parts=None, reasoning_parts=None):
        if method == "session/load":
            raise ACPSubprocessClientError("gone", reason="protocol")
        if method == "session/new":
            return {"sessionId": "Y"}
        return {}

    with patch.object(c, "_spawn", return_value=proc), \
         patch.object(c, "_request", side_effect=_request_fail):
        c._ensure_session(5.0)
    assert c.resume_failed == {"sid": "X", "reason": "protocol"}

    # Respawn: this time session/load of the CURRENT session (Y) succeeds.
    proc.poll.return_value = 1

    def _request_succeed(method, params, *, timeout_seconds, text_parts=None, reasoning_parts=None):
        if method == "session/load":
            return {}
        return {}

    with patch.object(c, "_spawn", return_value=proc), \
         patch.object(c, "_request", side_effect=_request_succeed):
        c._ensure_session(5.0)

    assert c.resumed is True
    assert c.resume_failed is None  # reset, not leaked from the prior failure


def test_response_usage_carries_acp_session_id_surviving_close():
    """Regression (gateway-path persistence race): the live ACP session id must
    ride on response.usage so conversation_loop can persist it for resume.

    Reading agent.client.acp_session_id *after* the turn is racy — the per-request
    client is closed at request_complete and close() nulls acp_session_id, so the
    next turn would session/new with no context (verified broken in E2E plan B,
    2026-05-29). The id captured on the response must survive close().
    """
    c = _client()

    def _fake_ensure(timeout_seconds):
        c.acp_session_id = "sess-LIVE-1"
        c._initialized = True

    def _fake_request(method, params, *, timeout_seconds, text_parts=None, reasoning_parts=None):
        if text_parts is not None:
            text_parts.append("hi")

    with patch.object(c, "_ensure_session", side_effect=_fake_ensure), \
         patch.object(c, "_request", side_effect=_fake_request):
        resp = c._create_chat_completion(
            model="claude-agent-acp",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert resp.usage.acp_session_id == "sess-LIVE-1"
    assert resp.choices[0].message.content == "hi"

    # close() resets the client attribute, but the captured response id is stable.
    c.close()
    assert c.acp_session_id is None
    assert resp.usage.acp_session_id == "sess-LIVE-1"


def test_requested_model_is_applied_once_before_prompt():
    c = _client(provider_label="kiro-acp")
    calls = []

    def _fake_ensure(timeout_seconds):
        c.acp_session_id = "sess-model"
        c._initialized = True

    def _fake_request(method, params, *, timeout_seconds, text_parts=None, reasoning_parts=None):
        calls.append((method, params))
        if text_parts is not None:
            text_parts.append("ok")

    with patch.object(c, "_ensure_session", side_effect=_fake_ensure), \
         patch.object(c, "_request", side_effect=_fake_request):
        for _ in range(2):
            c._create_chat_completion(
                model="claude-sonnet-4.5",
                messages=[{"role": "user", "content": "hello"}],
            )

    model_calls = [params for method, params in calls if method == "session/set_model"]
    assert model_calls == [{"sessionId": "sess-model", "modelId": "claude-sonnet-4.5"}]


def test_thought_chunk_goes_to_reasoning():
    c = _client()
    reasoning: list[str] = []
    c._consume_session_update(
        {"update": {"sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": "thinking"}}},
        None, reasoning,
    )
    assert "".join(reasoning) == "thinking"


# ── streaming generator ──────────────────────────────────────────────

def _streaming_client(inbox_msgs, *, session_id="sess-1"):
    """Client wired with a fake alive subprocess and a pre-filled inbox so
    _prompt_stream can be driven without a live adapter."""
    c = _client()
    c.acp_session_id = session_id
    c._next_id = 0  # → req_id becomes 1
    proc = MagicMock()
    proc.poll.return_value = None  # alive
    c._proc = proc
    for m in inbox_msgs:
        c._inbox.put(m)
    return c


def _upd(kind, **extra):
    return {"method": "session/update",
            "params": {"update": {"sessionUpdate": kind, **extra}}}


def test_prompt_stream_yields_content_reasoning_heartbeat_and_final():
    c = _streaming_client([
        _upd("agent_thought_chunk", content={"type": "text", "text": "pondering"}),
        _upd("tool_call", toolCallId="t1", title="Bash"),
        _upd("agent_message_chunk", content={"type": "text", "text": "Hello "}),
        _upd("agent_message_chunk", content={"type": "text", "text": "world"}),
        _upd("usage_update", used=100, size=1000, cost={"amount": 0.01}),
        {"id": 1, "result": {"stopReason": "completed"}},
    ])
    chunks = list(c._prompt_stream("hi", new_count=1, model="claude-agent-acp", timeout_seconds=5))

    content = "".join(
        ch.choices[0].delta.content for ch in chunks
        if ch.choices and ch.choices[0].delta.content
    )
    reasoning = "".join(
        ch.choices[0].delta.reasoning_content for ch in chunks
        if ch.choices and ch.choices[0].delta.reasoning_content
    )
    assert content == "Hello world"
    assert reasoning == "pondering"
    # tool_call + usage_update produce heartbeat chunks (empty choices) so the
    # consumer's idle watchdog resets while Claude works silently with tools.
    assert any(ch.choices == [] for ch in chunks)
    # final chunk carries finish_reason + usage with the live session id + cost.
    final = chunks[-1]
    assert final.choices[0].finish_reason == "stop"
    assert final.usage.acp_session_id == "sess-1"
    assert final.usage.cost_usd == pytest.approx(0.01)
    assert final.usage.context_used == 100
    assert c._delivered_count == 1


@pytest.mark.parametrize("stop_reason,expected", [
    ("completed", "stop"),
    ("end_turn", "stop"),
    ("max_tokens", "length"),
    ("maxTokens", "length"),
    ("cancelled", "stop"),
    ("weird", "stop"),
])
def test_prompt_stream_stop_reason_mapping(stop_reason, expected):
    c = _streaming_client([{"id": 1, "result": {"stopReason": stop_reason}}])
    chunks = list(c._prompt_stream("hi", 1, "m", 5))
    assert chunks[-1].choices[0].finish_reason == expected


def test_prompt_stream_missing_stop_reason_defaults_to_stop():
    c = _streaming_client([{"id": 1, "result": {}}])
    chunks = list(c._prompt_stream("hi", 1, "m", 5))
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_prompt_stream_rpc_error_raises_classified():
    c = _streaming_client([{"id": 1, "error": {"message": "Unauthorized"}}])
    with pytest.raises(ACPSubprocessClientError) as ei:
        list(c._prompt_stream("hi", 1, "m", 5))
    assert ei.value.reason == "auth"


def test_prompt_stream_raises_when_adapter_dies():
    c = _streaming_client([])
    c._proc.poll.return_value = 1  # exited
    with pytest.raises(ACPSubprocessClientError) as ei:
        list(c._prompt_stream("hi", 1, "m", 5))
    assert ei.value.reason in ("startup", "auth", "rate_limit", "unavailable")


def test_prompt_stream_idle_timeout():
    c = _streaming_client([])  # empty inbox, proc alive → idle
    with pytest.raises(ACPSubprocessClientError) as ei:
        list(c._prompt_stream("hi", 1, "m", timeout_seconds=0.2))
    assert ei.value.reason == "timeout"


# ── clean cancel (session/cancel) ────────────────────────────────────

def test_cancel_sends_session_cancel():
    c = _client()
    c.acp_session_id = "sess-9"
    proc = MagicMock()
    proc.poll.return_value = None
    c._proc = proc
    sent: list = []
    with patch.object(c, "_notify", side_effect=lambda m, p: sent.append((m, p))):
        c.cancel()
    assert sent == [("session/cancel", {"sessionId": "sess-9"})]


def test_cancel_noop_without_session_or_proc():
    c = _client()
    c.cancel()  # no proc, no session id — must not raise


def test_build_usage_flags_fresh_new_session():
    c = _client()
    c.resumed = False
    c.resume_failed = None
    u = c._build_usage(0.0)
    assert u.acp_resumed is False
    assert u.acp_resume_failed is None


def test_build_usage_flags_resumed():
    c = _client()
    c.resumed = True
    c.resume_failed = None
    u = c._build_usage(0.0)
    assert u.acp_resumed is True
    assert u.acp_resume_failed is None


def test_build_usage_flags_resume_failed():
    c = _client()
    c.resumed = False
    c.resume_failed = {"sid": "old-2", "reason": "unavailable"}
    u = c._build_usage(0.0)
    assert u.acp_resumed is False
    assert u.acp_resume_failed == {"sid": "old-2", "reason": "unavailable"}
    # Usage carries only the coarse resume flags — no extra seed metadata.
    assert not hasattr(u, "acp_seed_meta")


def test_streaming_and_blocking_usage_flags_match_on_resume_failure():
    """Streaming and non-streaming MUST surface identical resume flags — both
    paths build usage via _build_usage, so the honest signal is path-agnostic."""
    rf = {"sid": "old-3", "reason": "protocol"}

    def _make():
        c = _client()

        def _fake_ensure(timeout_seconds):
            c.acp_session_id = "sess-P"
            c._initialized = True
            c.resumed = False
            c.resume_failed = dict(rf)
        return c, _fake_ensure

    def _fake_request(method, params, *, timeout_seconds, text_parts=None, reasoning_parts=None):
        if text_parts is not None:
            text_parts.append("hi")

    msgs = [{"role": "user", "content": "hello"}]

    c1, ensure1 = _make()
    with patch.object(c1, "_ensure_session", side_effect=ensure1), \
         patch.object(c1, "_request", side_effect=_fake_request):
        blocking = c1._create_chat_completion(model="claude-agent-acp", messages=msgs)

    c2, ensure2 = _make()

    def _fake_prompt_stream(prompt_text, new_count, model, timeout_seconds, is_context_cmd=False):
        yield c2._final_chunk(model, "stop", c2._build_usage(0.0))

    with patch.object(c2, "_ensure_session", side_effect=ensure2), \
         patch.object(c2, "_prompt_stream", side_effect=_fake_prompt_stream):
        streaming_usage = list(c2._create_chat_completion(
            model="claude-agent-acp", stream=True, messages=msgs,
        ))[-1].usage

    assert blocking.usage.acp_resumed == streaming_usage.acp_resumed is False
    assert blocking.usage.acp_resume_failed == streaming_usage.acp_resume_failed == rf
