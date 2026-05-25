# Block 22 — Distance and 20% beam floor calibration

## What changed

- Beam power now starts at 20%, matching the mining laser UI behavior.
- The power slider in Streamlit now has `min_value=20` and defaults to `20`.
- `BeamState.power_percent` validation now rejects values below 20.
- Effective power now applies a distance-delivery factor:
  - closer rocks receive more delivered energy;
  - farther rocks receive less delivered energy;
  - the current reference point is 25m.
- Calculator notes now include `Distance efficiency` so the UI shows why effective power changed.
- Overpower is now treated as extra risk, so very high delivered power at close range is not always treated as safe.

## Why this exists

The previous approximation did not make short/medium distances visible enough in the result. A rock at 15m and the same rock at 45m could look almost identical unless distance exceeded 100m.

For real capture this was wrong enough to poison labels: the same beam percent can be underpowered at a longer distance and too strong at close range.

## Current heuristic

This is still not the official Star Citizen formula. It is a calibration heuristic:

```text
beam_effective_after_distance = beam_effective_before_distance * clamp(25 / distance, 0.05, 1.75)
```

The key behavior is:

```text
15m → stronger delivered power
45m → weaker delivered power
```

## MLOps relevance

This improves feature quality before real data collection. If the calculator ignores a feature that strongly affects gameplay, the saved labels become noisy:

```text
same rock + same power + different distance → different actual_outcome
```

The model needs this reflected in the generated features and rule-based baseline.
