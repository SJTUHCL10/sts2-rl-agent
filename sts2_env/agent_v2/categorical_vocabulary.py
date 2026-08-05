"""Collision-free categorical vocabulary for entity-v3 tensors."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

VOCABULARY_VERSION = "sts2-categorical-v1"
PAD_ID = 0
UNKNOWN_ID = 1
_ENCOUNTERED_UNKNOWN_TOKENS: set[str] = set()


def canonical_token(value: Any) -> str:
    """Canonicalize one categorical value without hashing it."""
    if value is None or value == "":
        return ""
    if hasattr(value, "name"):
        value = value.name
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    text = str(value).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").upper()


@dataclass(frozen=True, slots=True)
class CategoricalVocabulary:
    tokens: tuple[str, ...]
    token_to_id: dict[str, int]
    vocabulary_hash: str

    @classmethod
    def load_default(cls) -> "CategoricalVocabulary":
        resource = files("sts2_env.agent_v2").joinpath(
            "categorical_vocabulary_v1.json"
        )
        payload = json.loads(resource.read_text(encoding="utf-8"))
        if payload.get("version") != VOCABULARY_VERSION:
            raise ValueError(
                "Unsupported categorical vocabulary version: "
                f"{payload.get('version')!r}"
            )
        raw_tokens = [canonical_token(value) for value in payload["tokens"]]
        tokens = tuple(sorted(set(token for token in raw_tokens if token)))
        encoded = json.dumps(
            {"version": VOCABULARY_VERSION, "tokens": tokens},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        token_to_id = {
            token: index + 2 for index, token in enumerate(tokens)
        }
        return cls(
            tokens=tokens,
            token_to_id=token_to_id,
            vocabulary_hash=hashlib.sha256(encoded).hexdigest(),
        )

    @property
    def size(self) -> int:
        return len(self.tokens) + 2

    def encode(self, value: Any, *, strict: bool = False) -> int:
        token = canonical_token(value)
        if not token:
            return PAD_ID
        result = self.token_to_id.get(token)
        if result is not None:
            return result
        if strict:
            raise KeyError(f"Unknown categorical token: {token}")
        _ENCOUNTERED_UNKNOWN_TOKENS.add(token)
        return UNKNOWN_ID

    def contains(self, value: Any) -> bool:
        token = canonical_token(value)
        return not token or token in self.token_to_id


DEFAULT_CATEGORICAL_VOCABULARY = CategoricalVocabulary.load_default()
VOCABULARY_SIZE = DEFAULT_CATEGORICAL_VOCABULARY.size
VOCABULARY_HASH = DEFAULT_CATEGORICAL_VOCABULARY.vocabulary_hash


def categorical_id(value: Any, *, strict: bool = False) -> int:
    return DEFAULT_CATEGORICAL_VOCABULARY.encode(value, strict=strict)


def encountered_unknown_tokens(*, clear: bool = False) -> tuple[str, ...]:
    """Return raw canonical unknowns observed in this process."""
    result = tuple(sorted(_ENCOUNTERED_UNKNOWN_TOKENS))
    if clear:
        _ENCOUNTERED_UNKNOWN_TOKENS.clear()
    return result
