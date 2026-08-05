"""Exercise legal runs and detect unknown/overflow v3 categorical tokens."""

from __future__ import annotations

import argparse
import json

import numpy as np

from sts2_env.agent_v2.categorical_vocabulary import (
    UNKNOWN_ID,
    categorical_id,
    encountered_unknown_tokens,
)
from sts2_env.gym_env.entity_run_env import STS2EntityRunEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--fail-on-unknown", action="store_true")
    parser.add_argument("--stop-on-unknown", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    env = STS2EntityRunEnv()
    observation, _ = env.reset(seed=args.seed)
    categorical_values = 0
    unknown_values = 0
    overflow_observations = 0
    episodes = 0
    overflow_id = categorical_id("OVERFLOW", strict=True)

    completed_steps = 0
    for _ in range(args.steps):
        for key in (
            "global_categorical",
            "entity_categorical",
            "candidate_categorical",
        ):
            values = observation[key]
            categorical_values += int(np.count_nonzero(values))
            unknown_values += int(np.count_nonzero(values == UNKNOWN_ID))
        overflow_observations += int(np.count_nonzero(
            observation["entity_categorical"][:, 0] == overflow_id
        ))
        valid = np.flatnonzero(env.action_masks())
        action = int(rng.choice(valid))
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episodes += 1
            observation, _ = env.reset(seed=args.seed + episodes)
        completed_steps += 1
        if args.stop_on_unknown and unknown_values:
            break
    env.close()

    result = {
        "requested_steps": args.steps,
        "steps": completed_steps,
        "episodes": episodes,
        "categorical_values": categorical_values,
        "unknown_values": unknown_values,
        "unknown_rate": unknown_values / max(categorical_values, 1),
        "unknown_tokens": encountered_unknown_tokens(),
        "overflow_observations": overflow_observations,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.fail_on_unknown and unknown_values:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
