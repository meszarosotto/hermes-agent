"""A client that opened a brand-new ACP session must send the conversation.

Measured (agent.log, session 20261002_213524_d37e4a, three times in 48 h): a
Kiro turn went silent for 900 s, the stale-stream watchdog killed it, and
stream_error_cleanup closed the request client -- and with it the kiro-cli
process that held the ACP session. The continuation call got a new client,
which logged "ACP session/new cause=no_resume_sid". _compute_delta still sent
only the messages after the last assistant row: the bare "[System: The previous
response was cut off ...]" nudge, to a session that had never seen the
conversation. The model answered "I don't have any prior task content in this
session", and the turn ended as "Response truncated".

The background review takes the same path on every run. It builds its own
client, so the reviewer only ever received its instruction, never the
conversation it was asked to review ("result=none").

The delta is only valid for a session that already holds the transcript prefix:
one this client has prompted before, or one restored with session/load.
"""

from __future__ import annotations

from typing import Any

from agent.acp_subprocess_client import ACPSubprocessClient

NUDGE = (
    "[System: The previous response was cut off by a network error mid-stream. "
    "Continue the task from where you left off. Do not restart or repeat prior text.]"
)


class RecordingClient(ACPSubprocessClient):
    """The real client, with the adapter process replaced by a recorder.

    _ensure_session leaves the state the real one leaves after session/new
    (loaded=False) or a successful session/load (loaded=True); every
    session/prompt is recorded and answered at once.
    """

    def __init__(self, *, loaded: bool = False) -> None:
        super().__init__(acp_cwd="/tmp")
        self._loaded = loaded
        self.prompts: list[str] = []

    def _ensure_session(self, timeout_seconds: float) -> None:
        if self._initialized:
            return
        self.acp_session_id = "kiro-session"
        self.resumed = self._loaded
        self.resume_failed = None
        self._initialized = True
        self._delivered_count = 0

    def _request(self, method: str, params: dict[str, Any], **_: Any) -> Any:
        assert method == "session/prompt"
        self.prompts.append(params["prompt"][0]["text"])
        return {"stopReason": "end_turn"}


def transcript_after_a_killed_turn() -> list[dict[str, Any]]:
    """What _continue_text leaves in the transcript after a stale-stream kill."""
    return [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Find out why the jupyter-emesott pod restarts."},
        {"role": "assistant", "content": "The init container fails; checking its logs next."},
        {"role": "user", "content": "Check the roo-code-release-emesott namespace too."},
        {
            "role": "assistant",
            "content": "The pod there mounts ~/local as root, so",
            "_length_continuation_fragment": True,
        },
        {"role": "user", "content": NUDGE, "_length_continuation_nudge": True},
    ]


def test_a_replacement_client_sends_the_conversation_with_the_continuation():
    client = RecordingClient()  # what Hermes builds after stream_error_cleanup

    client._create_chat_completion(messages=transcript_after_a_killed_turn())

    prompt = client.prompts[0]
    assert "Find out why the jupyter-emesott pod restarts." in prompt
    assert "The init container fails; checking its logs next." in prompt
    assert "Check the roo-code-release-emesott namespace too." in prompt
    assert "The pod there mounts ~/local as root, so" in prompt  # text the user already saw
    assert prompt.rstrip().endswith(NUDGE)


def test_the_background_review_receives_the_conversation_it_reviews():
    client = RecordingClient()  # the review agent builds its own client
    transcript = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Release ericai-statusbar 1.3.0."},
        {"role": "assistant", "content": "Published to ARM, pipeline 9381439 green."},
        {"role": "user", "content": "Review the conversation above and update the skills."},
    ]

    client._create_chat_completion(messages=transcript)

    assert "Published to ARM, pipeline 9381439 green." in client.prompts[0]


def test_after_the_replay_only_new_turns_are_sent():
    client = RecordingClient()
    transcript = transcript_after_a_killed_turn()
    client._create_chat_completion(messages=transcript)

    transcript += [
        {"role": "assistant", "content": "...so the init container now chowns it."},
        {"role": "user", "content": "Ship it."},
    ]
    client._create_chat_completion(messages=transcript)

    assert client.prompts[1] == "Ship it."


def test_a_session_restored_with_session_load_gets_only_the_new_turn():
    client = RecordingClient(loaded=True)
    transcript = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]

    client._create_chat_completion(messages=transcript)

    assert client.prompts == ["second question"]


def test_a_new_conversation_starts_with_the_system_instructions():
    client = RecordingClient()

    client._create_chat_completion(messages=[
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Hello"},
    ])

    assert client.prompts == ["[System instructions]\nYou are Hermes.\n\nHello"]
