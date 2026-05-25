# Block 29 — Scan percent inputs

The UI now accepts Resistance and Instability as scan-style percent values.
Internally they are converted to normalized calculator fractions.

Examples:
- Scan Resistance `0%` -> `0.0`
- Scan Resistance `34%` -> `0.34`
- Scan Instability `18.54` -> `0.1854`
- Scan Instability `23` -> `0.23`

`RockInput` now validates resistance and instability as `0..1`, preventing
accidental use of raw percent values inside the formula.
