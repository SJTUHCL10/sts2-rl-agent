"""Episode-level JSONL metrics for long-running RL experiments."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from sts2_env.agent_v2.categorical_vocabulary import UNKNOWN_ID, categorical_id


class EpisodeMetricsCallback(BaseCallback):
    """Record every completed episode without depending on Monitor wrappers."""

    def __init__(self, output_path: Path, window: int = 100) -> None:
        super().__init__()
        self.output_path = output_path
        self.window = window
        self.episode_count = 0
        self.records: list[dict[str, Any]] = []
        self._returns: np.ndarray | None = None
        self._lengths: np.ndarray | None = None
        self._recent: deque[dict[str, Any]] = deque(maxlen=window)
        self._categorical_values = 0
        self._unknown_values = 0
        self._unknown_values_by_field: dict[str, int] = {}
        self._overflow_observations = 0
        self._combat_turn_limit_episodes = 0

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")
        self._returns = np.zeros(self.training_env.num_envs, dtype=np.float64)
        self._lengths = np.zeros(self.training_env.num_envs, dtype=np.int64)

    def _on_step(self) -> bool:
        assert self._returns is not None and self._lengths is not None
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        dones = np.asarray(self.locals["dones"], dtype=bool)
        infos = self.locals["infos"]
        new_obs = self.locals.get("new_obs")
        if isinstance(new_obs, dict):
            for key in (
                "global_categorical",
                "entity_categorical",
                "candidate_categorical",
            ):
                values = np.asarray(new_obs.get(key, ()))
                self._categorical_values += int(np.count_nonzero(values))
                self._unknown_values += int(np.count_nonzero(
                    values == UNKNOWN_ID
                ))
                if values.ndim > 0:
                    for field_index in range(values.shape[-1]):
                        field_unknowns = int(np.count_nonzero(
                            values[..., field_index] == UNKNOWN_ID
                        ))
                        if field_unknowns:
                            field_name = f"{key}[{field_index}]"
                            self._unknown_values_by_field[field_name] = (
                                self._unknown_values_by_field.get(
                                    field_name, 0
                                ) + field_unknowns
                            )
            entity_values = np.asarray(
                new_obs.get("entity_categorical", ())
            )
            if entity_values.ndim >= 2:
                overflow_id = categorical_id("OVERFLOW", strict=True)
                self._overflow_observations += int(np.count_nonzero(
                    entity_values[..., 0] == overflow_id
                ))
        self._returns += rewards
        self._lengths += 1
        for env_index in np.flatnonzero(dones):
            info = infos[int(env_index)]
            record = {
                "episode": self.episode_count,
                "timesteps": int(self.num_timesteps),
                "env_index": int(env_index),
                "return": float(self._returns[env_index]),
                "length": int(self._lengths[env_index]),
                "floor": int(info.get("floor", 0)),
                "act": int(info.get("act", 0)),
                "won": bool(info.get("player_won", False)),
                "truncated": bool(info.get("TimeLimit.truncated", False)),
                "combat_turn_limit_reached": bool(
                    info.get("combat_turn_limit_reached", False)
                ),
                "reward_components": info.get(
                    "episode_reward_components", {}
                ),
            }
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self.records.append(record)
            self._combat_turn_limit_episodes += int(
                record["combat_turn_limit_reached"]
            )
            self._recent.append(record)
            self.episode_count += 1
            self._returns[env_index] = 0.0
            self._lengths[env_index] = 0
        if self._recent:
            self.logger.record(
                "agent/mean_floor",
                float(np.mean([item["floor"] for item in self._recent])),
            )
            self.logger.record(
                "agent/mean_episode_return",
                float(np.mean([item["return"] for item in self._recent])),
            )
            self.logger.record(
                "agent/win_rate",
                float(np.mean([item["won"] for item in self._recent])),
            )
        self.logger.record(
            "agent/unknown_categorical_rate",
            self._unknown_values / max(self._categorical_values, 1),
        )
        self.logger.record(
            "agent/entity_overflow_observations",
            self._overflow_observations,
        )
        return True

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {"episodes": 0}
        return {
            "episodes": len(self.records),
            "wins": sum(bool(item["won"]) for item in self.records),
            "truncations": sum(
                bool(item["truncated"]) for item in self.records
            ),
            "mean_floor": float(np.mean([
                item["floor"] for item in self.records
            ])),
            "max_floor": max(item["floor"] for item in self.records),
            "mean_return": float(np.mean([
                item["return"] for item in self.records
            ])),
            "last_window_mean_floor": float(np.mean([
                item["floor"] for item in list(self._recent)
            ])),
            "last_window_mean_return": float(np.mean([
                item["return"] for item in list(self._recent)
            ])),
            "categorical_values": self._categorical_values,
            "unknown_categorical_values": self._unknown_values,
            "unknown_categorical_rate": (
                self._unknown_values / max(self._categorical_values, 1)
            ),
            "unknown_categorical_values_by_field": dict(
                sorted(self._unknown_values_by_field.items())
            ),
            "entity_overflow_observations": self._overflow_observations,
            "combat_turn_limit_episodes": self._combat_turn_limit_episodes,
        }
