# STS2 v0.111.0 spire-codex drift report

Source: `..\spire-codex\data-beta\v0.111.0\eng`

## Coverage

| Surface | Current data | Python coverage | Missing |
|---|---:|---:|---:|
| Cards: metadata | 595 | 595 | 0 |
| Cards: registered play behavior | 595 | 595 | 0 |
| Relics | 298 | 298 | 0 |
| Potions | 64 | 64 | 0 |
| Power IDs | 265 | 265 | 0 |
| Monsters (data inventory) | 115 | n/a | n/a |
| Encounters (data inventory) | 90 | n/a | n/a |

## Reference audits

- Card static metadata mismatches: 0
- Card dynamic variable mismatches: 0

## Missing IDs / behavior registrations

- Card behavior: None
- Relics: None
- Potions: None
- Powers: None

## Manual parity work still required

- Multiplayer-only targeting, ownership transfer, and cross-player card movement need bridge replay validation.
- Monster move effects and encounter composition are not inferred from JSON; use decompiled sources and scenario tests.
- Event option effects and run-level quest completion remain manual parity surfaces.
