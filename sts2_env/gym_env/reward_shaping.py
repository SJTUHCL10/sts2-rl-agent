"""Auditable, decomposed reward shaping for complete-run training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RunRewardShapingConfig:
    """Dense signals added to the unchanged terminal win/loss reward.

    The default step cost is paired with the v3 training default gamma=0.999:
    an indefinitely delayed -1 loss has approximately the same return as an
    immediate -1 loss, removing the old incentive to stall without progress.
    """

    floor_reward: float = 0.02
    combat_win_reward: float = 0.02
    hp_loss_penalty: float = 0.20
    step_penalty: float = 0.001

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def shaped_run_reward(
    base_reward: float,
    *,
    previous_floor: int,
    current_floor: int,
    previous_hp: int,
    current_hp: int,
    previous_max_hp: int,
    previous_phase: str,
    current_phase: str,
    run_over: bool,
    player_won: bool,
    config: RunRewardShapingConfig,
) -> tuple[float, dict[str, float]]:
    """Return total reward and named components for one environment step."""
    floor_progress = max(0, current_floor - previous_floor)
    hp_lost = max(0, previous_hp - current_hp)
    hp_scale = max(1, previous_max_hp)
    won_combat = (
        previous_phase == "COMBAT"
        and current_phase != "COMBAT"
        and (not run_over or player_won)
    )
    components = {
        "base": float(base_reward),
        "floor": float(floor_progress * config.floor_reward),
        "combat_win": float(config.combat_win_reward if won_combat else 0.0),
        "hp_loss": float(-config.hp_loss_penalty * hp_lost / hp_scale),
        "step": float(-config.step_penalty),
    }
    return float(sum(components.values())), components


def reward_component_sum(info: dict[str, Any]) -> float:
    components = info.get("reward_components", {})
    if not isinstance(components, dict):
        return 0.0
    return float(sum(float(value) for value in components.values()))
