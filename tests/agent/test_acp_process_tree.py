"""Closing an ACP client must also end what the adapter started.

kiro-cli runs the agent's tools itself (shell commands, builds, tests) as its
own child processes. Measured: a Kiro worker's `cargo test` hung; Hermes'
stale-stream kill closed the client, which terminated kiro-cli only. The test
processes were re-parented to launchd and kept running. The retried turn
started the same command again -- six stale kills later, six hung test runs
(with their fake agents and daemons) were still alive 1.5 hours after the
worker had given up.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from agent.acp_subprocess_client import ACPSubprocessClient

pytestmark = [
    pytest.mark.skipif(os.name == "nt", reason="POSIX process groups"),
    # Real signals to real processes: the adapter's process group, and the
    # cleanup of a tool process that an unfixed close() leaves re-parented to
    # launchd (outside the test's subtree, so the guard would refuse it).
    pytest.mark.live_system_guard_bypass,
]

FAKE_ADAPTER = r'''
import json, subprocess, sys
# A tool the agent started: still running when the client is closed.
tool = subprocess.Popen(["/bin/sleep", "60"])
with open(sys.argv[1], "w") as handle:
    handle.write(str(tool.pid))
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    result = {"sessionId": "s-1"} if msg["method"] == "session/new" else {"protocolVersion": 1}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\n")
    sys.stdout.flush()
'''


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # windows-footgun: ok - module skipped on Windows
    except ProcessLookupError:
        return False
    return True


def _gone_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def test_close_ends_the_tools_the_adapter_started(tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text(FAKE_ADAPTER, encoding="utf-8")
    pidfile = tmp_path / "tool.pid"
    client = ACPSubprocessClient(
        acp_command=sys.executable, acp_args=[str(adapter), str(pidfile)],
        acp_cwd=str(tmp_path), set_permission_mode=False,
    )
    tool_pid = None
    try:
        client._ensure_session(10)
        tool_pid = int(pidfile.read_text(encoding="utf-8-sig"))
        assert _alive(tool_pid)

        client.close()

        assert _gone_within(tool_pid, 5), "the adapter's tool process outlived close()"
    finally:
        if tool_pid and _alive(tool_pid):
            os.kill(tool_pid, getattr(signal, "SIGKILL", signal.SIGTERM))  # windows-footgun: ok
        client.close()


def test_the_adapter_does_not_share_the_callers_process_group(tmp_path):
    # Ctrl+C in the Hermes terminal goes to the foreground process group. An
    # adapter in that group dies with it and takes the live session along;
    # Hermes interrupts a turn with session/cancel instead.
    adapter = tmp_path / "adapter.py"
    adapter.write_text(FAKE_ADAPTER, encoding="utf-8")
    pidfile = tmp_path / "tool.pid"
    client = ACPSubprocessClient(
        acp_command=sys.executable, acp_args=[str(adapter), str(pidfile)],
        acp_cwd=str(tmp_path), set_permission_mode=False,
    )
    try:
        client._ensure_session(10)
        assert os.getpgid(client._proc.pid) != os.getpgid(0)  # windows-footgun: ok
    finally:
        client.close()
        if pidfile.exists():
            tool_pid = int(pidfile.read_text(encoding="utf-8-sig"))
            if _alive(tool_pid):
                os.kill(tool_pid, getattr(signal, "SIGKILL", signal.SIGTERM))  # windows-footgun: ok
