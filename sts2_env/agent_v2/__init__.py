"""Versioned entity/candidate interface for the next-generation agent."""

from sts2_env.agent_v2.candidates import ActionCandidate, build_action_candidates
from sts2_env.agent_v2.schema import (
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    schema_manifest,
)
from sts2_env.agent_v2.snapshot import (
    attach_v2_envelope,
    build_combat_snapshot,
    build_run_snapshot,
    build_run_decision_snapshot,
)
from sts2_env.agent_v2.tensorizer import (
    DEFAULT_TENSORIZER_CONFIG,
    TENSOR_ENCODING_VERSION,
    TensorizerConfig,
    observation_space,
    tensorize_snapshot,
)

__all__ = [
    "ACTION_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "ActionCandidate",
    "attach_v2_envelope",
    "build_action_candidates",
    "build_combat_snapshot",
    "build_run_snapshot",
    "build_run_decision_snapshot",
    "schema_manifest",
    "DEFAULT_TENSORIZER_CONFIG",
    "TENSOR_ENCODING_VERSION",
    "TensorizerConfig",
    "observation_space",
    "tensorize_snapshot",
]
