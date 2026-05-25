# Block 21 — Build presets for every mining ship

This block keeps a single general calculator page and adds starter build presets for every ship currently represented in the project:

- Prospector
- MOLE
- Golem

The goal is fast real-data capture: select a ship, select a loadout preset, enter rock/resource/outcome fields, and save the event.

## Added preset families

Each ship now has four starter presets:

1. `2x Rieger-C3`
2. `Rieger-C3 + FLTR`
3. `Rieger-C3 + Torrent`
4. `Rieger-C3 + Torrent III`

These are configuration presets, not final balance truth. Module coefficients are still calibration approximations.

## MLOps relevance

The selected build is saved into every event as `build_id` and `ship_type`. That lets later analytics and ML answer questions like:

- which build has more `good` outcomes;
- which build often produces `too_slow` or `not_enough_power`;
- whether a build changes yield/profit behavior;
- whether the model should learn build-specific patterns.

## UI behavior

The sidebar now has two controls:

1. `Ship`
2. `Build profile`

The selected loadout is shown in the sidebar so the operator can verify head/module setup before saving events.
