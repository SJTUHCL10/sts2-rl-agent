from __future__ import annotations

import copy

import numpy as np
import pytest

import sts2_env.agent_v2.tensorizer as tensorizer_module
from sts2_env.agent_v2.tensorizer import (
    ENTITY_TYPE_TO_ID,
    TensorizerConfig,
    observation_space,
    tensorize_snapshot,
)
from sts2_env.agent_v2.snapshot import build_run_decision_snapshot
from sts2_env.gym_env.entity_run_env import STS2EntityRunEnv
from sts2_env.run.run_manager import RunManager


def test_entity_run_env_observation_matches_declared_space() -> None:
    config = TensorizerConfig(max_entities=128)
    env = STS2EntityRunEnv(
        tensorizer_config=config,
        max_steps=20,
    )
    observation, info = env.reset(seed=109)

    assert env.observation_space.contains(observation)
    assert observation["candidate_categorical"].shape == (157, 7)
    assert observation["candidate_numeric"].shape == (157, 15)
    assert observation["entity_mask"].sum() > 0
    valid = np.flatnonzero(info["action_mask"])
    next_observation, _, _, _, _ = env.step(int(valid[0]))
    assert env.observation_space.contains(next_observation)


def test_entity_run_env_reuses_info_action_mask(monkeypatch) -> None:
    env = STS2EntityRunEnv(max_steps=20)
    original = env.run_env.action_masks
    calls = 0

    def counted_action_masks() -> np.ndarray:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(env.run_env, "action_masks", counted_action_masks)
    _, info = env.reset(seed=109)

    assert calls == 1
    assert env.action_masks() is info["action_mask"]
    env._structured_observation()
    assert calls == 1

    env.step(int(np.flatnonzero(info["action_mask"])[0]))
    assert calls == 2
    env.action_masks()
    assert calls == 2

    env.invalidate_action_mask_cache()
    env.action_masks()
    assert calls == 3


def test_entity_run_env_skips_discarded_legacy_observation(monkeypatch) -> None:
    env = STS2EntityRunEnv(max_steps=20)

    def fail_if_called() -> np.ndarray:
        raise AssertionError("entity wrapper should not encode the v1 vector")

    monkeypatch.setattr(env.run_env, "_encode_obs", fail_if_called)
    observation, info = env.reset(seed=109)
    env.step(int(np.flatnonzero(info["action_mask"])[0]))

    assert env.observation_space.contains(observation)


def test_entity_run_env_returns_structured_terminal_observation() -> None:
    config = TensorizerConfig(max_entities=64)
    env = STS2EntityRunEnv(tensorizer_config=config, max_steps=20)
    env.reset(seed=109)
    assert env.run_env._mgr is not None
    env.run_env._mgr.run_state.lose_run()
    env.invalidate_action_mask_cache()

    observation = env._structured_observation()

    assert env.observation_space.contains(observation)
    assert env.run_env.entity_observation()["phase"] == "RUN_OVER"


def test_identical_deck_cards_are_count_compressed() -> None:
    config = TensorizerConfig(max_entities=32)
    card = {
        "entity_id": "card:player:0:template",
        "card_id": "CLONE",
        "owner_id": "player:0",
        "zone": "deck",
        "zone_index": 0,
        "card_type": "SKILL",
        "upgrade_level": 0,
        "afflictions": {},
        "enchantments": {},
    }
    cards = []
    for index in range(10_000):
        duplicate = dict(card)
        duplicate["entity_id"] = f"card:player:0:{index}"
        duplicate["zone_index"] = index
        cards.append(duplicate)
    snapshot = {
        "type": "map_select",
        "phase": "MAP_CHOICE",
        "run_state": {
            "character_id": "Ironclad",
            "players": [{"entity_id": "player:0", "hp": 80, "max_hp": 80}],
            "cards": cards,
        },
        "candidates": [],
    }
    mask = np.zeros(157, dtype=np.int8)
    mask[115] = 1

    observation = tensorize_snapshot(snapshot, mask, config, validate=True)
    card_rows = (
        observation["entity_categorical"][:, 0]
        == ENTITY_TYPE_TO_ID["CARD"]
    )

    assert card_rows.sum() == 1
    count_feature = observation["entity_numeric"][card_rows, 12]
    assert count_feature.item() > 1.0
    assert observation_space(config).contains(observation)


def test_full_observation_validation_is_opt_in(monkeypatch) -> None:
    config = TensorizerConfig(max_entities=8)
    mask = np.zeros(157, dtype=np.int8)
    mask[0] = 1
    snapshot = {
        "type": "run_complete",
        "phase": "RUN_OVER",
        "candidates": [],
    }

    class RejectingSpace:
        def contains(self, _value) -> bool:
            return False

    monkeypatch.setattr(
        tensorizer_module,
        "observation_space",
        lambda _config: RejectingSpace(),
    )

    tensorize_snapshot(snapshot, mask, config)
    with pytest.raises(ValueError, match="violates its Gymnasium space"):
        tensorize_snapshot(snapshot, mask, config, validate=True)


def test_combat_piles_compress_but_hand_instances_do_not() -> None:
    config = TensorizerConfig(max_entities=32)
    cards = [
        {
            "entity_id": f"card:player:0:draw:{index}",
            "card_id": "CLONE",
            "owner_id": "player:0",
            "zone": "draw",
            "zone_index": index,
            "combat_vars": {},
        }
        for index in range(1000)
    ]
    cards.extend([
        {
            "entity_id": f"card:player:0:hand:{index}",
            "card_id": "CLONE",
            "owner_id": "player:0",
            "zone": "hand",
            "zone_index": index,
            "combat_vars": {},
        }
        for index in range(2)
    ])
    snapshot = {
        "type": "combat_action",
        "phase": "COMBAT",
        "cards": cards,
        "candidates": [],
    }
    mask = np.zeros(157, dtype=np.int8)
    mask[0] = 1

    observation = tensorize_snapshot(snapshot, mask, config)
    card_rows = (
        observation["entity_categorical"][:, 0]
        == ENTITY_TYPE_TO_ID["CARD"]
    )

    assert card_rows.sum() == 3
    counts = observation["entity_numeric"][card_rows, 12]
    assert sum(count > 1.0 for count in counts) == 1


def test_candidate_uses_stable_content_not_runtime_instance_id() -> None:
    config = TensorizerConfig(max_entities=16)
    mask = np.zeros(157, dtype=np.int8)
    mask[1] = 1

    def make_snapshot(instance: str) -> dict:
        entity_id = f"card:player:0:{instance}"
        return {
            "type": "combat_action",
            "phase": "COMBAT",
            "cards": [{
                "entity_id": entity_id,
                "card_id": "STRIKE",
                "owner_id": "player:0",
                "zone": "hand",
                "zone_index": 0,
            }],
            "candidates": [{
                "candidate_id": f"combat:play:{entity_id}:none",
                "action_type": "PLAY_CARD",
                "source_id": entity_id,
                "enabled": True,
                "payload": {
                    "action": "play",
                    "card_index": 0,
                    "target_index": -1,
                },
            }],
        }

    first = tensorize_snapshot(make_snapshot("100"), mask, config)
    second = tensorize_snapshot(make_snapshot("999999"), mask, config)

    np.testing.assert_array_equal(
        first["candidate_categorical"][1],
        second["candidate_categorical"][1],
    )


def test_card_selection_state_changes_candidate_features() -> None:
    config = TensorizerConfig(max_entities=16)
    mask = np.zeros(157, dtype=np.int8)
    mask[0:3] = 1
    base = {
        "type": "card_select",
        "phase": "CARD_REWARD",
        "min_select": 1,
        "max_select": 2,
        "selected_count": 0,
        "can_confirm": False,
        "cards": [
            {"index": 0, "id": "STRIKE", "type": "Attack", "selected": False},
            {"index": 1, "id": "DEFEND", "type": "Skill", "selected": False},
        ],
    }
    from sts2_env.agent_v2.candidates import build_action_candidates

    base["candidates"] = [
        item.to_dict() for item in build_action_candidates(base)
    ]
    unselected = tensorize_snapshot(base, mask, config)

    selected_state = copy.deepcopy(base)
    selected_state["cards"][0]["selected"] = True
    selected_state["selected_count"] = 1
    selected_state["can_confirm"] = True
    selected_state.pop("candidates")
    selected_state["candidates"] = [
        item.to_dict() for item in build_action_candidates(selected_state)
    ]
    selected = tensorize_snapshot(selected_state, mask, config)

    assert unselected["candidate_numeric"][1, 10] == 0.0
    assert selected["candidate_numeric"][1, 10] == 1.0
    assert selected["candidate_numeric"][0, 14] == 1.0

    from sts2_env.agent_v2.categorical_vocabulary import categorical_id

    # The candidate itself is linked to STRIKE, rather than relying on the
    # arbitrary action-slot embedding to identify which card it selects.
    assert selected["candidate_categorical"][1, 2] == categorical_id(
        "STRIKE", strict=True
    )
    assert selected["candidate_categorical"][1, 5] == categorical_id(
        "STRIKE", strict=True
    )


def test_run_snapshot_preserves_incremental_deck_selection() -> None:
    manager = RunManager(seed=803, character_id="Ironclad")
    assert manager.run_state.player.obtain_relic("KIFUDA")

    before = build_run_decision_snapshot(manager)
    assert before["type"] == "card_select"
    assert before["selected_count"] == 0
    assert before["can_confirm"] is True
    assert not any(card["selected"] for card in before["cards"])

    manager.take_action({"action": "choose", "index": 0})
    after = build_run_decision_snapshot(manager)

    assert after["selected_count"] == 1
    assert after["cards"][0]["selected"] is True
    selected_candidate = next(
        candidate for candidate in after["candidates"]
        if candidate["payload"] == {"action": "choose", "index": 0}
    )
    assert selected_candidate["features"]["selected"] is True


def test_typed_set_encoder_is_entity_permutation_invariant() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from sts2_env.models.typed_set_transformer import (
        TypedSetTransformerConfig,
        TypedSetTransformerExtractor,
    )

    config = TensorizerConfig(max_entities=96)
    env = STS2EntityRunEnv(tensorizer_config=config, max_steps=20)
    observation, _ = env.reset(seed=9)
    permuted = copy.deepcopy(observation)
    permutation = np.random.default_rng(1).permutation(config.max_entities)
    for key in ("entity_categorical", "entity_numeric", "entity_mask"):
        permuted[key] = permuted[key][permutation]

    extractor = TypedSetTransformerExtractor(
        env.observation_space,
        config=TypedSetTransformerConfig(
            d_model=32,
            num_heads=4,
            num_inducing_points=8,
            num_memory_tokens=4,
            num_isab_layers=1,
        ),
        categorical_vocab_size=config.categorical_vocab_size,
    )
    extractor.eval()

    def tensors(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(value).unsqueeze(0)
            for key, value in obs.items()
        }

    with torch.no_grad():
        original_features = extractor(tensors(observation))
        permuted_features = extractor(tensors(permuted))
    torch.testing.assert_close(
        original_features,
        permuted_features,
        atol=1e-5,
        rtol=1e-5,
    )


def test_maskable_policy_forward_shapes() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from sts2_env.models.typed_set_transformer import (
        TypedSetMaskableActorCriticPolicy,
        TypedSetTransformerConfig,
    )

    config = TensorizerConfig(max_entities=64)
    env = STS2EntityRunEnv(tensorizer_config=config, max_steps=20)
    observation, _ = env.reset(seed=5)
    policy = TypedSetMaskableActorCriticPolicy(
        env.observation_space,
        env.action_space,
        lambda _: 3e-4,
        typed_set_config=TypedSetTransformerConfig(
            d_model=32,
            num_heads=4,
            num_inducing_points=8,
            num_memory_tokens=4,
            num_isab_layers=1,
        ),
        categorical_vocab_size=config.categorical_vocab_size,
        categorical_vocabulary_hash=config.categorical_vocabulary_hash,
        ortho_init=False,
    )
    batched = {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in observation.items()
    }
    actions, values, log_prob = policy(
        batched,
        action_masks=env.action_masks()[None, :],
    )

    assert actions.shape == (1,)
    assert values.shape == (1, 1)
    assert log_prob.shape == (1,)
    assert env.action_masks()[int(actions.item())] == 1


def test_maskable_ppo_learns_saves_and_loads(tmp_path) -> None:
    pytest.importorskip("torch")
    sb3_contrib = pytest.importorskip("sb3_contrib")
    from sts2_env.models.typed_set_transformer import (
        TypedSetMaskableActorCriticPolicy,
        TypedSetTransformerConfig,
    )

    config = TensorizerConfig(max_entities=64)
    env = STS2EntityRunEnv(tensorizer_config=config, max_steps=20)
    model = sb3_contrib.MaskablePPO(
        TypedSetMaskableActorCriticPolicy,
        env,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "typed_set_config": TypedSetTransformerConfig(
                d_model=16,
                num_heads=4,
                num_inducing_points=4,
                num_memory_tokens=2,
                num_isab_layers=1,
            ),
            "categorical_vocab_size": config.categorical_vocab_size,
            "categorical_vocabulary_hash": config.categorical_vocabulary_hash,
            "tensorizer_layout_hash": config.feature_layout_hash(),
            "ortho_init": False,
        },
        seed=1,
        verbose=0,
    )

    model.learn(total_timesteps=8)
    model_path = tmp_path / "typed_set_model"
    model.save(model_path)
    loaded = sb3_contrib.MaskablePPO.load(model_path, env=env)
    observation, _ = env.reset(seed=2)
    action, _ = loaded.predict(
        observation,
        action_masks=env.action_masks(),
        deterministic=True,
    )

    assert env.action_masks()[int(action)] == 1
