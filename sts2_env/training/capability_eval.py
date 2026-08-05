"""Fixed-seed evaluation and best-checkpoint selection for run agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True, slots=True)
class CapabilityScoreConfig:
    """Weights for a capability score used only to select checkpoints."""

    win_weight: float = 100.0
    truncation_penalty: float = 10.0

    def score(self, metrics: dict[str, Any]) -> float:
        episodes = max(int(metrics.get("episodes", 0)), 1)
        truncation_rate = (
            float(metrics.get("truncated_episodes", 0)) / episodes
        )
        return (
            self.win_weight * float(metrics.get("win_rate", 0.0))
            + float(metrics.get("mean_floor", 0.0))
            - self.truncation_penalty * truncation_rate
        )


class CapabilityEvalCallback(BaseCallback):
    """Evaluate on fixed seeds and save the best capability checkpoint.

    SB3's default evaluation callback selects by mean reward. Shaped rewards
    and long multi-select loops make that a poor proxy for STS2 ability, so
    this callback explicitly combines wins, floor progress, and truncations.
    """

    def __init__(
        self,
        *,
        eval_freq: int,
        evaluation_fn: Callable[[Any], dict[str, Any]],
        output_dir: Path,
        score_config: CapabilityScoreConfig | None = None,
    ) -> None:
        super().__init__()
        self.eval_freq = max(1, int(eval_freq))
        self.evaluation_fn = evaluation_fn
        self.output_dir = output_dir
        self.score_config = score_config or CapabilityScoreConfig()
        self.best_score = float("-inf")
        self.best_metrics: dict[str, Any] | None = None
        self.evaluations: list[dict[str, Any]] = []
        self.log_path = output_dir / "capability_evaluations.jsonl"

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        self._evaluate_and_maybe_save()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        self._evaluate_and_maybe_save()
        return True

    def _evaluate_and_maybe_save(self) -> None:
        metrics = dict(self.evaluation_fn(self.model))
        score = self.score_config.score(metrics)
        record = {
            "timesteps": int(self.num_timesteps),
            "capability_score": score,
            **metrics,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.evaluations.append(record)
        self.logger.record("eval_capability/score", score)
        self.logger.record(
            "eval_capability/mean_floor",
            float(metrics.get("mean_floor", 0.0)),
        )
        self.logger.record(
            "eval_capability/win_rate",
            float(metrics.get("win_rate", 0.0)),
        )
        self.logger.record(
            "eval_capability/truncated_episodes",
            int(metrics.get("truncated_episodes", 0)),
        )
        if score > self.best_score:
            self.best_score = score
            self.best_metrics = record
            best_dir = self.output_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(best_dir / "best_model")
            (best_dir / "evaluation.json").write_text(
                json.dumps(record, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def summary(self) -> dict[str, Any]:
        return {
            "evaluations": len(self.evaluations),
            "best_score": (
                self.best_score if self.best_metrics is not None else None
            ),
            "best": self.best_metrics,
            "score_config": {
                "win_weight": self.score_config.win_weight,
                "truncation_penalty": self.score_config.truncation_penalty,
            },
        }
