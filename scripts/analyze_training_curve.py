"""Summarize episode-level JSONL metrics from an agent training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def summarize_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return decision-relevant statistics for one timestep bucket."""
    floors = [int(record["floor"]) for record in records]
    returns = [float(record["return"]) for record in records]
    lengths = [int(record["length"]) for record in records]
    return {
        "episodes": len(records),
        "mean_floor": float(np.mean(floors)),
        "max_floor": max(floors),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "wins": sum(bool(record["won"]) for record in records),
        "truncations": sum(
            bool(record["truncated"]) for record in records
        ),
    }


def summarize_curve(
    records: list[dict[str, Any]], bucket_steps: int
) -> dict[str, Any]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        bucket = int(record["timesteps"]) // bucket_steps
        buckets.setdefault(bucket, []).append(record)
    return {
        "episodes": len(records),
        "bucket_steps": bucket_steps,
        "buckets": [
            {
                "start_step": bucket * bucket_steps,
                "end_step": (bucket + 1) * bucket_steps,
                **summarize_bucket(buckets[bucket]),
            }
            for bucket in sorted(buckets)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("curve_path", type=Path)
    parser.add_argument("--bucket-steps", type=int, default=10_000)
    args = parser.parse_args()
    if args.bucket_steps <= 0:
        parser.error("--bucket-steps must be positive")
    records = [
        json.loads(line)
        for line in args.curve_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(json.dumps(
        summarize_curve(records, args.bucket_steps),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
