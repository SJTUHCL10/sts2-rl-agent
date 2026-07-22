# STS2 Advisor Mod

`STS2AdvisorMod` is a display-only companion for a trained full-run policy.
It observes actionable game states and shows a recommendation, while the
player keeps complete control of every decision and input.

The first MVP supports:

- combat card, target, potion, and end-turn recommendations;
- reachable map-node recommendations;
- card-reward picks or skips;
- stale-response rejection when the player acts before inference finishes;
- indefinite idle waiting while the player reads or plans;
- an always-on-top text panel, toggled with `F8`.

It intentionally contains no AutoSlay startup, input simulation, card command,
or UI click calls. The existing automatic Bridge Mod is not a dependency and
should normally be disabled while testing the advisor.

## Build and install

The project reuses `bridge_mod/STS2BridgeMod.local.props` when a dedicated
`advisor_mod/AdvisorMod.local.props` is absent, so the existing local Steam,
Godot, and BaseLib paths work without duplication.

```powershell
cd C:\Users\A\Codes\sts\sts2-rl-agent
C:\Users\A\Codes\sts\.tools\dotnet-sdk-9\dotnet.exe build advisor_mod\AdvisorMod.csproj
```

The build installs `STS2AdvisorMod.dll`, `STS2AdvisorMod.pck`,
`STS2AdvisorMod.json`, and `mod_manifest.json` into the game's
`mods/STS2AdvisorMod/` directory. Enable BaseLib and STS2 Advisor in the mod
launcher. Do not enable `STS2BridgeMod` for a normal manual-advisor run because
that separate mod launches AutoSlay.

## Run the model service

Activate the shared environment and start the runner from the repository root:

```powershell
conda activate C:\Users\A\Codes\sts\.conda\sts
python -m sts2_env.bridge.advisor_runner
```

The default checkpoint is:

```text
output/full_run_v0109_smoke_masked_20260721/final_model.zip
```

The mod listens on loopback TCP port `9003`; the automatic Bridge uses `9002`,
so the two protocols do not collide. A different model or port can be selected
with `--model-path` and `--port`.

The displayed policy action is a preference from the smoke-test model, not a
win probability. This checkpoint is useful for protocol and UI validation but
its recorded evaluation was 0% full-run wins, so recommendation quality should
not yet be treated as strong gameplay guidance.

The current 151-element full-run observation does not explicitly encode reward
card identities, enchantment IDs, or affliction IDs. The simulator's combat
observation has card ID, cost, damage, block, and type slots, but the current
real-game serializer only supplies ID, cost, type, target, and playability, so
its damage/block slots remain zero. Enchantment identity is not a separate
feature. The advisor protocol now preserves modifier metadata for display and
future adapters; teaching the policy to use it properly requires observation
parity work, an observation-layout change for modifier identity, and retraining.

## Next coverage

The Python side already uses `FullRunStateAdapter`; extending the C# collector
with existing Bridge-compatible states will add rest sites, shops, events,
treasure, boss relics, card bundles, and card-selection overlays without
changing the policy interface.
