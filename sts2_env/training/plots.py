"""Static training plots generated from durable JSONL metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read valid JSON objects, tolerating a partial final line."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _rolling_mean(values: list[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return array
    size = min(max(int(window), 1), len(array))
    cumulative = np.cumsum(np.insert(array, 0, 0.0))
    result = (cumulative[size:] - cumulative[:-size]) / size
    return np.concatenate((np.full(size - 1, np.nan), result))


def render_training_plots(
    output_dir: Path,
    *,
    window: int = 100,
) -> list[Path]:
    """Render episode and fixed-seed evaluation PNGs when data is available."""
    matplotlib_config = output_dir / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config.resolve()))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    episodes = read_jsonl(output_dir / "training_curve.jsonl")
    if episodes:
        steps = [int(item.get("timesteps", 0)) for item in episodes]
        floor = [float(item.get("floor", 0.0)) for item in episodes]
        returns = [float(item.get("return", 0.0)) for item in episodes]
        lengths = [float(item.get("length", 0.0)) for item in episodes]
        wins = [float(bool(item.get("won", False))) for item in episodes]

        figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        series = (
            (axes[0, 0], floor, "Floor", "Floor"),
            (axes[0, 1], returns, "Episode return", "Return"),
            (axes[1, 0], lengths, "Episode length", "Length"),
            (axes[1, 1], wins, "Win rate", "Win indicator"),
        )
        for axis, values, title, label in series:
            axis.scatter(steps, values, s=5, alpha=0.12, color="#4C78A8")
            axis.plot(
                steps,
                _rolling_mean(values, window),
                linewidth=2,
                color="#F58518",
                label=f"rolling {min(window, len(values))}",
            )
            axis.set_title(title)
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            axis.legend(loc="best")
        axes[1, 0].set_xlabel("Training timesteps")
        axes[1, 1].set_xlabel("Training timesteps")
        figure.suptitle(f"Training progress ({len(episodes)} episodes)")
        figure.tight_layout()
        path = output_dir / "training_progress.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)

    evaluations = read_jsonl(output_dir / "capability_evaluations.jsonl")
    if evaluations:
        steps = [int(item.get("timesteps", 0)) for item in evaluations]
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        plots = (
            (axes[0, 0], "mean_floor", "Mean floor"),
            (axes[0, 1], "max_floor", "Max floor"),
            (axes[1, 0], "capability_score", "Capability score"),
            (axes[1, 1], "win_rate", "Win rate"),
        )
        for axis, key, title in plots:
            axis.plot(
                steps,
                [float(item.get(key, 0.0)) for item in evaluations],
                marker="o",
                linewidth=2,
                color="#54A24B",
            )
            axis.set_title(title)
            axis.grid(alpha=0.25)
        axes[1, 0].set_xlabel("Training timesteps")
        axes[1, 1].set_xlabel("Training timesteps")
        figure.suptitle("Fixed-seed capability evaluation")
        figure.tight_layout()
        path = output_dir / "capability_progress.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)
    return created
