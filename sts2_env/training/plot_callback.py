"""SB3 callback for periodically refreshing static training plots."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from sts2_env.training.plots import render_training_plots


class TrainingPlotCallback(BaseCallback):
    """Refresh PNG summaries while training and once more at shutdown."""

    def __init__(self, output_dir: Path, *, plot_freq: int, window: int) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.plot_freq = max(1, int(plot_freq))
        self.window = max(1, int(window))

    def _on_step(self) -> bool:
        if self.n_calls % self.plot_freq == 0:
            render_training_plots(self.output_dir, window=self.window)
        return True

    def _on_training_end(self) -> None:
        render_training_plots(self.output_dir, window=self.window)
