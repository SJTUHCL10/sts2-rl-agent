from __future__ import annotations

from sts2_env.agent_v2.schema import schema_manifest


def test_schema_manifest_cache_does_not_share_mutable_results() -> None:
    first = schema_manifest()
    expected_cards = first["vocabulary_sizes"]["cards"]
    first["vocabulary_sizes"]["cards"] = -1

    second = schema_manifest()

    assert second["vocabulary_sizes"]["cards"] == expected_cards
