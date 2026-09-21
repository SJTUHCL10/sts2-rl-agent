"""Regenerate static plots for an entity-v2 training output directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from sts2_env.training.plots import render_training_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    for path in render_training_plots(args.output_dir, window=args.window):
        print(path)


if __name__ == "__main__":
    main()
