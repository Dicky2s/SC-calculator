# Block 31 — High instability scan values

Star Citizen can show instability values above 100 on scan, for example
`Instability 284.74` for impossible rocks.

Changes:
- UI `Instability, %` allows values up to 1000.
- Percent normalization no longer clamps upper bound.
- `284.74` is stored/calculated as normalized `2.8474`.
- `RockInput.instability` allows values above 1.
- Data quality sanity max for normalized instability is relaxed to 10.
