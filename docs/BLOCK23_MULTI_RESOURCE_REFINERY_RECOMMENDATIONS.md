# Block 23 — Multi-resource capture, refinery block, power/distance helper

## What changed

1. Resource capture is now a dynamic table.
   - A single mining event can store several resource rows.
   - Each row contains resource name, resource percent, raw SCU estimate, and a comment.
   - The legacy summary fields `primary_resource`, `resource_percent`, and `raw_scu_estimate` are still filled for compatibility.

2. Refinery data moved into a separate optional block.
   - Refinery method, location, future complete time, actual refined SCU, fees, and sell value can be captured later.
   - This keeps mining capture fast while leaving a place for delayed refinery outcomes.

3. Added a power/distance helper.
   - It scans 10–120m and 20–100% power for the current rock/build.
   - It shows a minimum warm-up pair and a recommended stable-hold pair.
   - It uses the same rule-based calculator, so results are heuristic and should be calibrated with real outcomes.

## MLOps purpose

This block improves data quality before real collection:

- Multi-resource rows preserve mixed-rock composition.
- Refinery fields make future profit/yield modeling possible.
- Power/distance helper adds a repeatable rule-based recommendation that can later be compared against real outcomes.

The important future targets are:

- `actual_outcome` for good/not-good classification.
- `refined_scu_actual`, `sell_value_auec`, and resource rows for profit/yield modeling.
- power/distance recommendation errors for formula calibration.
