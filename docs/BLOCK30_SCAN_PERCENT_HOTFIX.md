# Block 30 — Scan percent hotfix

Fixes a UI wiring bug where scan-style Instability/Resistance values such as
30 could still reach `RockInput` as `30.0` instead of normalized `0.30`.

Changes:
- UI fields are `Resistance, %` and `Instability, %`.
- UI converts these values via `scan_percent_to_fraction`.
- Default scan distance is now 15m instead of 92m.
- `RockInput` keeps strict `0..1` validation for formula safety.
