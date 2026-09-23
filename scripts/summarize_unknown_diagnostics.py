"""Summarize per-worker UNKNOWN token JSONL from an entity training run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(directory: Path) -> list[dict]:
    counts: Counter[tuple[str, str]] = Counter()
    samples: dict[tuple[str, str], dict] = {}
    for path in sorted(directory.glob("worker_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            key = (record["field"], record["token"])
            counts[key] += int(record["count"])
            samples.setdefault(key, record["sample"])
    return [
        {"field": field, "token": token, "count": count, "sample": samples[(field, token)]}
        for (field, token), count in counts.most_common()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
