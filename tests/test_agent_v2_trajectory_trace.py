"""Readable trajectory recording and offline viewer tests."""

from __future__ import annotations

import json

import numpy as np

from scripts.trace_agent_v2 import trace_episode
from sts2_env.agent_v2.tensorizer import TensorizerConfig
from sts2_env.agent_v2.snapshot import build_run_decision_snapshot
from sts2_env.agent_v2.trajectory_trace import TRACE_FORMAT, compact_state, describe_action, write_trace
from sts2_env.core.enums import RoomType
from sts2_env.events.act2 import CrystalSphere
from sts2_env.run.run_manager import RunManager
from sts2_env.potions.base import create_potion


def test_random_episode_records_each_decision_and_state(tmp_path) -> None:
    trace = trace_episode(
        model=None,
        config=TensorizerConfig(),
        seed=109,
        deterministic=True,
        max_steps=8,
        max_combat_turns=50,
    )
    assert trace["format"] == TRACE_FORMAT
    assert trace["summary"]["steps"] == len(trace["steps"]) == 8
    assert trace["summary"]["truncated"]
    assert [step["step"] for step in trace["steps"]] == list(range(1, 9))
    assert all(step["action"]["slot"] >= 0 for step in trace["steps"])
    assert all("hp" in step["before"] and "hp" in step["after"] for step in trace["steps"])
    assert all("reward_components" in step for step in trace["steps"])

    json_path, html_path = write_trace(trace, tmp_path / "seed_109")
    assert json.loads(json_path.read_text(encoding="utf-8"))["steps"] == trace["steps"]
    html = html_path.read_text(encoding="utf-8")
    assert "中文" in html and "English" in html
    assert 'id="trace-data"' in html
    assert "seed 109" in html
    assert "for(const e of state.enemies)" in html
    assert "state.choices && state.choices.length" in html
    assert "const goldDelta=" in html


def test_viewer_escapes_embedded_html_script(tmp_path) -> None:
    trace = {
        "seed": 1, "model_path": "</script><script>bad()</script>",
        "summary": {"floor": 1, "steps": 0, "won": False}, "steps": [],
    }
    _, html_path = write_trace(trace, tmp_path / "unsafe")
    html = html_path.read_text(encoding="utf-8")
    assert "</script><script>bad()" not in html
    assert "\\u003c/script>" in html


def test_compact_state_records_event_and_reward_choices() -> None:
    manager = RunManager(seed=3, character_id="Ironclad")
    manager._phase = RunManager.PHASE_EVENT
    manager._current_room_type = RoomType.EVENT
    manager._event_model = CrystalSphere()
    state = compact_state({
        "phase": "EVENT",
        "options": [{"id": "leave", "label": "Leave", "action": "event_choice"}],
        "run_state": {"floor": 3, "players": [{"hp": 70, "max_hp": 80, "gold": 99}]},
    }, manager)
    assert state["room_type"] == "EVENT"
    assert state["event_id"] == "CrystalSphere"
    assert state["choices"][0]["id"] == "leave"

    manager._phase = RunManager.PHASE_CARD_REWARD
    reward = compact_state({
        "phase": "CARD_REWARD",
        "cards": [{"id": "STRIKE", "cost": 1}, {"id": "DEFEND", "cost": 1}],
    }, manager)
    assert [choice["id"] for choice in reward["choices"]] == ["STRIKE", "DEFEND"]


def test_legacy_trace_labels_fixed_reward_but_flags_old_missing_input() -> None:
    manager = RunManager(seed=4, character_id="Ironclad")
    manager._phase = RunManager.PHASE_CARD_REWARD
    manager._offered_potion = create_potion("FruitJuice")
    snapshot = build_run_decision_snapshot(manager)
    mask = np.zeros(285, dtype=np.int8)
    mask[120] = mask[123] = 1

    described = describe_action(snapshot, mask, 120, legacy_v5=True)

    assert described["action_type"] == "PICK_POTION"
    assert described["source"] == "FruitJuice"
    assert not described.get("candidate_missing")
    assert described["legacy_model_candidate_missing"]
