"""Tests for bridge client protocol helpers."""

from __future__ import annotations

import json

from sts2_env.bridge.protocol import BridgeAction
from sts2_env.bridge.client import STS2GameClient


class FakeSocket:
    def __init__(self):
        self.sent: list[bytes] = []

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)


def _last_payload(client: STS2GameClient) -> dict:
    assert client._sock is not None  # noqa: SLF001
    raw = client._sock.sent[-1].decode("utf-8").strip()  # noqa: SLF001
    return json.loads(raw)


def test_send_action_echoes_last_request_id():
    client = STS2GameClient()
    client._sock = FakeSocket()  # noqa: SLF001
    client._connected = True  # noqa: SLF001
    client._last_request_id = "abc123"  # noqa: SLF001

    client.send_action({"action": BridgeAction.END_TURN})

    assert _last_payload(client) == {"action": BridgeAction.END_TURN, "request_id": "abc123"}
    assert client._last_request_id is None  # noqa: SLF001


def test_send_action_preserves_explicit_request_id():
    client = STS2GameClient()
    client._sock = FakeSocket()  # noqa: SLF001
    client._connected = True  # noqa: SLF001
    client._last_request_id = "stale"  # noqa: SLF001

    client.send_action({"action": BridgeAction.CHOOSE, "index": 2, "request_id": "fresh"})

    assert _last_payload(client) == {"action": BridgeAction.CHOOSE, "index": 2, "request_id": "fresh"}


def test_choose_candidate_echoes_request_and_decision_ids():
    client = STS2GameClient()
    client._sock = FakeSocket()  # noqa: SLF001
    client._connected = True  # noqa: SLF001
    client._last_request_id = "request-7"  # noqa: SLF001
    client._last_decision_id = "decision-7"  # noqa: SLF001

    client.choose_candidate("combat:end_turn")

    assert _last_payload(client) == {
        "action": BridgeAction.CANDIDATE,
        "candidate_id": "combat:end_turn",
        "decision_id": "decision-7",
        "request_id": "request-7",
    }


def test_choose_many_and_skip_helpers():
    client = STS2GameClient()
    client._sock = FakeSocket()  # noqa: SLF001
    client._connected = True  # noqa: SLF001

    client.choose_many([0, 2])
    assert _last_payload(client) == {"action": BridgeAction.CHOOSE, "indexes": [0, 2]}

    client.skip()
    assert _last_payload(client) == {"action": BridgeAction.SKIP}
