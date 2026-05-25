# Real ML runbook

This runbook is for the **manual real-data** model only.

Synthetic datasets and synthetic models are smoke-test artifacts. They validate the pipeline, but they are not gameplay advice.

## Required input

The source of truth is:

```text
data/sessions/manual_events.jsonl
```

Each useful training row needs:

```text
actual_outcome != unknown
```

Minimum practical start:

```text
30-50 labeled events
```

Better target:

```text
100+ labeled events
```

The dataset should include both binary classes:

```text
good
not_good = bad / too_slow / too_unstable / not_enough_power / overheated / wrong_prediction
```

## UI flow

1. Save events in **Calculator**.
2. Later update outcomes in **Saved events → Outcome labeling queue**.
3. Open **Saved events → Real ML run starter**.
4. Keep minimum labeled rows at `30+` for a real weak baseline.
5. Click **Run real ML pipeline**.
6. Inspect:
   - dataset path
   - quality status
   - training readiness
   - training result
   - promotion gate result
7. Promote only when the gate passes and the data is real enough.

## CLI flow

From project root:

```powershell
python scripts/run_real_ml_pipeline.py --min-labeled 30
```

Run export/readiness only:

```powershell
python scripts/run_real_ml_pipeline.py --no-train
```

Train and promote if gate passes:

```powershell
python scripts/run_real_ml_pipeline.py --min-labeled 30 --promote-if-passed
```

For a smoke test only, you may lower the threshold:

```powershell
python scripts/run_real_ml_pipeline.py --min-labeled 6 --min-test-rows 1 --min-accuracy 0.0
```

Do not treat a low-threshold model as production-like.

## Output artifacts

```text
data/datasets/mining_events.csv
models/mining_outcome_baseline_manual.joblib
reports/baseline_model_report_manual.json
reports/training_runs.jsonl
models/active_model.json
```

## Production-like principle

The project keeps the pipeline explicit:

```text
raw events
  → labeled dataset
  → quality report
  → training readiness
  → manual model artifact
  → training report
  → training run history
  → promotion gate
  → active model pointer
  → prediction logging
  → post-inference evaluation
```

That is the local equivalent of a small MLOps lifecycle.
