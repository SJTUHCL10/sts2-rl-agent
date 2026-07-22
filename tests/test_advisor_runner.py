from sts2_env.bridge.advisor_runner import describe_recommendation


def test_describes_combat_card_and_target() -> None:
    state = {
        "type": "combat_action",
        "hand": [{"id": "STRIKE"}],
        "enemies": [{"id": "CULTIST"}],
    }
    summary, details = describe_recommendation(
        state,
        {"action": "play", "card_index": 0, "target_index": 0},
        7,
    )
    assert summary == "打出手牌 #1 STRIKE → #1 CULTIST"
    assert "#7" in details


def test_describes_map_choice_by_protocol_index() -> None:
    state = {
        "type": "map_select",
        "nodes": [{"index": 3, "type": "Elite", "row": 2, "col": 1}],
    }
    summary, _ = describe_recommendation(
        state, {"action": "choose", "index": 3}, 121
    )
    assert "Elite" in summary
    assert "第 3 行" in summary


def test_describes_card_reward_skip() -> None:
    summary, _ = describe_recommendation(
        {"type": "card_reward"}, {"action": "skip"}, 119
    )
    assert summary == "跳过"


def test_describes_card_and_target_indexes_with_modifier_metadata() -> None:
    state = {
        "type": "combat_action",
        "hand": [{"id": "STRIKE", "enchantment": {"id": "Sharp", "amount": 3}}],
        "enemies": [
            {"id": "SLIME", "hp": 8, "max_hp": 20, "intent": "ATTACK"},
        ],
    }
    summary, _ = describe_recommendation(
        state,
        {"action": "play", "card_index": 0, "target_index": 0},
        12,
    )
    assert "手牌 #1 STRIKE [Sharp +3]" in summary
    assert "#1 SLIME（HP 8/20, ATTACK）" in summary


def test_client_can_disable_idle_timeout() -> None:
    from sts2_env.bridge.client import STS2GameClient

    assert STS2GameClient(timeout=None).timeout is None
