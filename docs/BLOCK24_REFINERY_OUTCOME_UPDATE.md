# Block 24 — Refinery outcome update

Adds a post-mining refinery update workflow.

## Why

Mining capture and refinery/sale results often happen at different times. The UI now supports:

1. Save the rock event immediately.
2. Later select the same event in **Saved events → Refinery outcome queue**.
3. Add refinery method, location, fee, total refined SCU, sell value, and per-resource refined outputs.

## Data fields

New/expanded fields:

- `refined_resource_count`
- `refined_resource_names`
- `total_refined_scu_actual`
- `total_resource_sell_value_auec`
- `refined_resources_json`

The original total fields remain:

- `refined_scu_actual`
- `refined_value_auec`
- `refinery_fee_auec`
- `sell_value_auec`

## MLOps value

This separates fast event capture from delayed label/yield enrichment:

```text
raw mining event → later refinery update → richer analytics/ML dataset
```

This enables future targets such as:

- profitable / not profitable
- expected refined SCU
- expected sell value
- per-resource yield quality
