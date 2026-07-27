# Bridge Replay Harness

This harness adds a lightweight golden-comparison loop for parity work.

It is intended for this workflow:

1. Run the real game through the TCP bridge.
2. Record bridge states and chosen actions into a replay JSON file.
3. Recreate the same combat in the simulator.
4. Replay the recorded actions and compare each resulting state snapshot.

The current comparator supports:

- combat-only traces through `CombatState`
- partial run traces through `RunManager`

Supported bridge message types:

- `combat_action`
- `card_select`
- `map_select`
- `reward_screen`
- `card_bundle`
- `crystal_sphere`
- `card_reward`
- `rest_site`
- `shop`
- `event`
- `treasure`
- `boss_relic`
- `game_over`
- `run_complete`

The C# bridge mod now routes ordinary reward-screen, card-bundle,
Crystal Sphere, rest-site, shop, event, treasure, and boss-relic choices through
these same state types instead of using AutoSlay's random room handlers. Those
live messages can be recorded by the replay recorder.

The bridge wiring is guarded by
`tests/test_bridge_autoslay_coverage.py`, which checks that the main run rooms
and standalone choice screens stay attached to RL bridge handlers. Deck
upgrade, transform, enchant, simple-grid, deck-card, and choose-a-card flows
still keep their default AutoSlay screen handlers registered, but normal
`CardSelectCmd` paths are expected to be intercepted by `RlCardSelector` before
those screens handle the choice.

It still does not compare full-run traces end to end.

## Location

The harness lives in:

- [`sts2_env/parity/bridge_replay.py`](../sts2_env/parity/bridge_replay.py)

Key exports:

- `BridgeReplayRecorder`
- `BridgeReplayTrace`
- `save_replay_trace()`
- `load_replay_trace()`
- `combat_state_to_bridge_state()`
- `run_manager_to_bridge_state()`
- `compare_combat_replay()`
- `compare_run_replay()`

## Replay Format

Replay files are JSON with this structure:

```json
{
  "version": 1,
  "mode": "combat",
  "metadata": {
    "scenario_factory": "my_module:make_combat"
  },
  "initial_state": { "...": "bridge-like state" },
  "steps": [
    {
      "action": { "action": "play", "card_index": 0, "target_index": 0 },
      "resulting_state": { "...": "bridge-like state" }
    }
  ]
}
```

`initial_state` is the first actionable state seen from the real game.

Each step stores:

- the action sent to the game
- the next state observed after that action resolves

## Recording From a Real Bridge Session

Wrap the existing bridge client:

```python
from sts2_env.bridge.client import STS2GameClient
from sts2_env.parity.bridge_replay import BridgeReplayRecorder

client = STS2GameClient()
client.connect()
recorder = BridgeReplayRecorder(
    client,
    metadata={"scenario_factory": "my_module:make_combat"},
)

state = recorder.receive_state()
recorder.play_card(0, 0)
state = recorder.receive_state()

recorder.save("artifacts/my_trace.json")
```

The recorder only persists states relevant to the comparator:

- `combat_action`
- `card_select`
- `map_select`
- `reward_screen`
- `card_bundle`
- `crystal_sphere`
- `card_reward`
- `rest_site`
- `shop`
- `event`
- `treasure`
- `boss_relic`
- `game_over`
- `run_complete`

You can also record directly from the existing bridge runner:

```bash
python -m sts2_env.bridge.agent_runner \
  --model-path models/combat_ppo.zip \
  --record-replay artifacts/run_trace.json \
  --replay-factory my_module:make_run_manager
```

The recorded trace will be written when the run ends or when the runner exits.

### Resume an existing save for a focused recording

Bridge resume mode is deliberately fail-closed and disabled by default. Create
`STS2BridgeMod.runtime.json` beside the installed Bridge DLL:

```json
{
  "resume_existing_run": true
}
```

On the next modded launch, the Bridge clicks the native Continue button,
attaches its handlers to the restored `RunState`, drains an already-open
Crystal Sphere or reward overlay, and continues from the restored room or map.
If no valid run save exists, it reports an error and does not start a new run.
Set the field to `false` or remove the file to restore the normal behavior that
abandons an old run and starts a fresh Ironclad run.

For a focused Crystal Sphere trace, save before entering the event or while its
screen is open, enable resume mode, start the Python recorder near the modded
game launch, and stop it after the event returns to the map.

## Comparing Against the Simulator

Build a deterministic factory that recreates the same combat setup:

```python
from sts2_env.parity.bridge_replay import compare_combat_replay

result = compare_combat_replay(
    "artifacts/my_trace.json",
    factory="my_module:make_combat",
)

if not result.success:
    for mismatch in result.mismatches:
        print(mismatch)
```

For run-level traces:

```python
from sts2_env.parity.bridge_replay import compare_run_replay

result = compare_run_replay(
    "artifacts/map_or_reward_trace.json",
    factory="my_module:make_run_manager",
)
```

There is also a CLI helper:

```bash
python -m sts2_env.parity.bridge_replay_cli show artifacts/run_trace.json

python -m sts2_env.parity.bridge_replay_cli compare \
  artifacts/run_trace.json \
  --mode run \
  --factory my_module:make_run_manager
```

`compare_combat_replay()`:

- compares the simulator's initial state to the recorded initial state
- replays each recorded action in order
- supports recorded card plays, turn ends, card-select choices, skips, and potion use
- compares each resulting simulator state to the recorded state
- stops at the first mismatch and returns a diff list

## What Gets Compared

For `combat_action`, the comparator checks:

- player HP / max HP / block / energy / max energy / powers
- hand order and card info (`id`, `cost`, `type`, `target`, `playable`, `upgraded`, `base_damage`, `base_block`)
- enemy order and combat info (`id`, `hp`, `max_hp`, `block`, `is_alive`, `intent`, `intent_damage`, `intent_hits`, `powers`)
- draw / discard / exhaust counts
- round number

For `card_select`, it checks:

- card list
- `min_select`
- `max_select`

For `map_select`, it checks:

- node order
- node type
- row / col
- floor
- act

For `card_reward`, it checks:

- offered card list
- card order and metadata
- `can_skip`

For `card_bundle`, it checks:

- bundle order
- cards inside each bundle
- stable action / enabled status
- floor
- act

For `rest_site`, it checks:

- option order
- option id / action / label / enabled status
- floor
- act

For `reward_screen`, `shop`, `event`, `treasure`, and `boss_relic`, it checks:

- option order
- stable action / enabled status
- floor
- act

For `crystal_sphere`, new v2 recordings additionally check:

- option order, coordinates, affected-cell IDs, and enabled status;
- grid dimensions, tool, remaining divinations, and completion flags;
- every public cell's hidden/clickable/revealed-item state;
- revealed item identity, type, size, rarity/amount, and reveal order.

Legacy recordings without `minigame` retain the older option-only comparison.

## Current Limitations

- No full-run replay comparison yet.
- Run comparison currently targets actionable slices and can record terminal `game_over` / `run_complete` states, but it is not complete run lifecycle proof.
- Reward-screen, shop, event, treasure, and boss-relic comparison intentionally ignores localized labels and simulator-only option ids; live C# labels and Python internal ids do not always use the same naming scheme.
- Crystal Sphere v2 comparison covers the public logical board. UI animation timing, save/quit, multiplayer message ordering, and exact reward identity still need dedicated live traces.
- The AutoSlay wiring guard proves that key handlers are registered, but it is
  not proof that every original UI path is reachable or intercepted correctly in
  a live client.
- Event-triggered combat now routes through the same bridge combat handler as ordinary combat, but it still needs live-game smoke testing.
- C# bridge handler compilation and live-game smoke testing still require a local `dotnet`/STS2 mod build environment.
- The trace must be paired with a deterministic simulator factory that recreates the same combat or run setup.

This is deliberate: the goal is to give parity work a reusable combat golden harness now, without blocking on a full run-state serializer.
