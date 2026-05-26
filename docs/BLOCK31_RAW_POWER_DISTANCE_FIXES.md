# Block 31 — raw power, distance validation, and resource aggregation fixes

## Goal

This block fixes the calculator calibration issues found during manual Prospector testing:

- `19635 mass / 0 resistance / 15 m / Helix S1 + 2x Rieger-C3` should require about `80%` power, not `20%`.
- Mining distance must be treated as a hard head-range constraint. For the current heads in this project, `45 m` is the max range. A captured value like `88 m` is invalid and should not be calculated as a valid mining event.
- Distance efficiency must never boost beam power above `1.0`.
- Resource rows with duplicate names are aggregated, and `Raw SCU` is derived from `Composition total SCU * scan percent / 100` when the UI value is missing or left at zero.

## Formula changes

The calculator now uses raw mining power units.

```text
required_power = mass * 0.2 / (1 - resistance)
```

`resistance` is stored as a fraction:

```text
47% -> 0.47
```

For a beam:

```text
full_power = head.max_power * passive_module_multipliers * active_module_multipliers
beam_power_before_distance = full_power * power_percent / 100
effective_power = beam_power_before_distance * distance_efficiency
```

## Distance rules

Per head:

```text
distance <= optimal_range -> distance_efficiency = 1.0
optimal_range < distance <= max_range -> linear falloff toward 0.35 at max range
distance > max_range -> ValueError / invalid event
```

Current configs set:

```text
optimal_range = 15 m
max_range = 45 m
```

## Calibration example

For `Prospector + Helix S1 + 2x Rieger-C3`:

```text
Helix S1 max_power = 3150
Rieger-C3 = x1.25
Rieger-C3 = x1.25
full_power = 3150 * 1.25 * 1.25 = 4921.875
```

For a rock:

```text
mass = 19635
resistance = 0
required_power = 19635 * 0.2 = 3927
required_input = 3927 / 4921.875 * 100 = 79.79%
```

Expected result:

```text
20% -> need_more_power
80% -> edge_take
90%+ -> take
```

## ML/data quality changes

- `formula_issue_flag=True` rows are excluded from supervised training.
- Distance sanity max is now `45.0` in dataset quality checks.
- Invalid distances are rejected at calculation time.

## Tests

The test suite was updated to use valid mining distances and to assert the new raw-power behavior.

```text
132 passed
```
