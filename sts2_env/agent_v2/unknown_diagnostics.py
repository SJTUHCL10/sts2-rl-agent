"""Bounded, per-worker diagnostics for categorical vocabulary misses."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from sts2_env.agent_v2.categorical_vocabulary import canonical_token


class UnknownTokenDiagnostics:
    """Aggregate misses without writing one log line per padded token."""

    def __init__(
        self,
        output_path: Path,
        *,
        worker_index: int,
        flush_observations: int = 256,
    ) -> None:
        self.output_path = output_path
        self.worker_index = worker_index
        self.flush_observations = flush_observations
        self.observations = 0
        self._counts: Counter[tuple[str, str]] = Counter()
        self._samples: dict[tuple[str, str], dict[str, Any]] = {}

    def record(
        self,
        field: str,
        value: Any,
        context: dict[str, Any],
    ) -> None:
        token = canonical_token(value)
        key = (field, token)
        self._counts[key] += 1
        if key not in self._samples:
            self._samples[key] = {
                **context,
                "raw_value": str(value)[:160],
                "observation": self.observations,
            }

    def note_observation(self) -> None:
        self.observations += 1
        if self.observations % self.flush_observations == 0:
            self.flush()

    def flush(self) -> None:
        if not self._counts:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            for (field, token), count in sorted(self._counts.items()):
                handle.write(json.dumps({
                    "worker_index": self.worker_index,
                    "field": field,
                    "token": token,
                    "count": count,
                    "sample": self._samples[(field, token)],
                }, ensure_ascii=False, sort_keys=True) + "\n")
        self._counts.clear()
        self._samples.clear()
