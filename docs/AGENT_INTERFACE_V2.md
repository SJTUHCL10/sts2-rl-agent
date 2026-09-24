# Agent Interface v2

The first neural consumer of this interface is documented in
[Typed Set Transformer Agent v2](TYPED_SET_TRANSFORMER_AGENT.md).

This document defines the Phase 0/1 engineering contract for the next agent.
It deliberately stops before choosing or implementing a neural-network
architecture.

## Compatibility boundary

The existing interfaces are frozen:

- combat v1: 131 observations, 115 discrete actions;
- full-run v1: 151 observations, 157 discrete actions.

Existing checkpoints continue to use those layouts. V2 is additive and does
not reinterpret any v1 slot. A v2 checkpoint must store and validate:

- `protocol_version`;
- `observation_schema`;
- `action_schema`;
- `feature_layout_hash`;
- card, power, and relic vocabulary hashes.

The canonical manifest is returned by
`sts2_env.agent_v2.schema.schema_manifest()`. Replay files record the same
manifest. Python tests lock the shared C# constants to this manifest.

## Observation contract

V2 transports typed entity sets rather than a pre-flattened tensor:

- global run and phase fields;
- all players;
- combat creatures, powers, and full intent lists;
- card instances from deck, hand, draw, discard, and exhaust piles;
- relic instances and their observable state;
- potion slots;
- map nodes and directed edges;
- currently legal semantic action candidates.

Transport coverage is not model-input coverage. The current
`typed-set-tensor-v8` projection consumes map nodes and their coordinates,
type, visited/reachable flags, but does **not** encode `map_edges`; it cannot
reconstruct connected routes from the set of nodes alone. It also reduces a
structured entity `counters` map to a numeric sum. Cards now encode one
affliction type/amount and one enchantment type/amount; this corrects the v5
simulator affliction omission and enchantment-amount loss. Conversely, each
creature's current intents
(up to three types with damage/hits), hand-card instances, and action-candidate
source/target entity pointers do reach the model. Current v8 training starts
with Neow and exposes the chosen Act 1 variant
(`Overgrowth` or `Underdocks`) as a global categorical feature. Act-specific
event and encounter pools follow the selected variant. The historical v5 1M run
recorded zero UNKNOWN categorical values and zero entity overflow, but is not
compatible with v8 inference. The v6 500k checkpoint predates the Crystal
Sphere UNKNOWN fix and can only be inspected with the explicit legacy-v6
trace projection. See
[Typed Set Transformer Agent v2](TYPED_SET_TRANSFORMER_AGENT.md) for the exact
projection; none of these tensor limits changes the v2 wire contract.
The current Python simulator tracks an affliction's type but not a variable
stack amount, so its card snapshots emit amount `1`; the live game-side v2
serializer exposes the actual amount when available.

The current simulator now registers built-in events in plain training processes,
and v6 snapshots add a visible event-ID entity and semantic option IDs. Potion
and relic reward picks likewise have a choice entity and a candidate source-row
pointer. The explicit legacy-v5 tensor projection omits these additions so old
checkpoints can be diagnosed without silently changing their inputs. The prior
v5 training run also double-applied Burning Blood after victories and saw empty
event rooms; its metrics are not a valid post-fix baseline.
For live Bridge event messages, the option's `event_id` is used to create the
same event-context row when available. Live option labels are localized text,
not stable simulator option IDs; parity of option-specific policy behavior
still needs live Bridge validation.

Each entity has a stable `entity_id` within a decision snapshot. Card instances
include identity, owner, zone, zone index, current and original cost, type,
target type, upgrade state, enchantments, afflictions, and dynamic variables.
Game-side dynamic variables retain `base`, `enchanted`, and player-visible
`preview` values.

Power entities additionally carry `power_type`, `stack_type`, `amount`,
`display_amount`, `amount_on_turn_start`, `skip_next_duration_tick`,
`is_visible`, `dynamic_vars`, and `counters`. Relic entities carry `rarity`,
`stack_count`, `status`, `is_used_up`, `is_melted`, `is_wax`,
`show_counter`, `display_amount`, `floor_added`, `dynamic_vars`, and
`counters`. The Python projection includes JSON-safe simulator counters such
as Girya lifts and Nunchaku/Pen Nib attack counts; the game projection uses
the authoritative public `DisplayAmount`, `ShowCounter`, status, and dynamic
variables.

Crystal Sphere is a first-class entity decision rather than an opaque event
option list. Its snapshot contains the 11x11 public fog grid, remaining
divinations, selected tool, completion/placement flags, fully revealed items,
and one coordinate-stable candidate per clickable cell. Hidden cells never
carry item identity. `affected_cell_ids` follows the game's exact Big-tool
order: horizontal, vertical, diagonal, then the clicked cell.

Absence is distinct from numeric zero. During transient game screens the C#
collector may set `run_state_available=false`; consumers must not replace that
with a fabricated zero-valued run.

The simulator exposes the same representation through:

```python
combat_env.entity_observation()
run_env.entity_observation()
```

The old `reset()` and `step()` observations remain unchanged.

## Action contract

V2 sends a variable-length `candidates` array. A candidate contains:

```json
{
  "candidate_id": "combat:play:card:hand:0:enemy:0",
  "action_type": "PLAY_CARD",
  "source_id": "card:hand:0",
  "target_id": "enemy:0",
  "enabled": true,
  "features": {},
  "payload": {
    "action": "play",
    "card_index": 0,
    "target_index": 0
  }
}
```

The v2 contract describes candidates semantically. The current Python policy
scores them through a transitional 285-slot padded action space (the first
157 slots retain legacy meanings), rather than a fully dynamic distribution.
The Python client responds with:

```json
{
  "action": "candidate",
  "candidate_id": "combat:play:card:hand:0:enemy:0",
  "request_id": "42",
  "decision_id": "42"
}
```

The shared C# protocol layer validates that the candidate is present and
enabled, then converts its stored payload to the existing handler command.
This keeps the already-tested game-action code unchanged. Legacy action
messages remain supported.

Candidate IDs describe semantic entities, so reordering display items does not
change action identity. Candidate ordering is nevertheless deterministic and
must be preserved from observation through masking and selection.

Card selectors with `min_select=0` expose `card_select:skip`. The current
candidate-v2 contract represents one atomic choice per decision. Existing
legacy `choose_many(indexes)` remains available for selectors that require
multiple cards; a learned policy for those screens needs either an
autoregressive selector loop or a future explicitly versioned multi-select
action extension. Simulator snapshots preserve `selected`, `selected_count`,
`min_select`, `max_select`, and `can_confirm`, so incremental toggle/confirm is
Markov and does not require recurrent policy state. The current game-side
`RlCardSelector` still expects a complete multi-index response for multi-select
screens; live incremental parity remains future work.

## Bridge and Advisor flow

`shared_mod/ProtocolV2.cs` is compiled into both mods and is the single
game-side enrichment and candidate-resolution implementation.

```text
game state collector
  -> frozen legacy state fields
  -> ProtocolV2 full run snapshot + entity IDs + candidates
  -> newline-delimited JSON
  -> EntityStateAdapter / future entity encoder
```

The automatic Bridge resolves candidate actions before dispatching them to its
existing handlers. The Advisor only sends enriched observations and remains
strictly passive: no click, command, or game-action path was added.

## Parity and validation

Required checks before training a v2 model:

1. Compare simulator and recorded real-game snapshots at the entity-field
   level, not only inventory counts.
2. Verify every enabled candidate resolves to the intended legacy command.
3. Record new v2 replays; old recordings permanently lack newly added fields.
4. Validate the manifest before loading a checkpoint or replay.
5. Live-smoke combat, card rewards, map, events, shop, rest, treasure,
   boss-relic, card-bundle, Crystal Sphere, and run completion.

### Crystal Sphere audit boundary

The Python implementation is pinned to the STS2 `0.110.0` decompilation:

- event cost is `50 + NextInt(1, 50)` with an exclusive upper bound;
- pay/debt branches start 3/6 divinations and Debt is added before play;
- corner clearing, 15-item population order/sizes, placement candidate order,
  retry/short-circuit behavior, and .NET-compatible RNG consumption match
  `CrystalSphereMinigame` and `CrystalSphereItem`;
- Big-tool affected-cell order and fully-clear reveal condition match the game;
- Doubt is added immediately on curse reveal; good rewards are materialized in
  reveal order using the event RNG.

Source-contract tests and fixed-seed golden layouts detect implementation or
decompilation drift. This is source-level parity evidence, not absolute
end-to-end proof: exact live potion/card/relic identities, reward-screen
lifecycle, save/quit, multiplayer synchronization, and UI timing still require
a fresh v2 Bridge recording from the same game build. The replay normalizer now
compares the public grid and revealed-item state when those fields are present;
old v1 recordings cannot supply that evidence.

Phase 0/1.1 provides the transport, compatibility, counter fields, and Crystal
Sphere simulation foundation. The model encoder, phase-specific policy heads,
training algorithm, batching rules, and multi-select policy are the next phase.
