# SC Mining Assistant

Manual baseline calculator for Star Citizen mining.

Current scope:

- YAML-based mining head/module/build configs
- Prospector and MOLE manual build profiles
- deterministic calculation core
- manual event logging to JSONL
- manual outcome labeling for saved events
- event dataset reader
- dataset export from JSONL to CSV
- dataset quality report
- basic analytics dashboard for formula-vs-outcome inspection
- baseline ML training for good vs not-good outcome
- Streamlit UI with:
  - calculator tab
  - saved events tab
  - actual outcome selector and comment field
  - filters by session, ship, verdict, actual outcome
  - verdict distribution chart
  - actual outcome distribution chart
  - numeric dataset summary
  - CSV export block
  - dataset quality report
  - basic analytics dashboard
  - baseline ML training block

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Run tests

```bash
python -m pytest -q
```

Expected:

```text
37 passed
```

## Run UI

```bash
python -m streamlit run src/sc_mining/ui/streamlit_app.py
```

## Data path

Manual UI events are written to:

```text
data/sessions/manual_events.jsonl
```

The file is JSONL: one calculation event per line.

Exported analytics/ML dataset is written to:

```text
data/datasets/mining_events.csv
```

## Event structure

Each saved event contains:

```text
session_id
build
rock
beams
result
outcome
```

`result` is the rule-based calculator output.

`outcome` is the manual label from the real in-game result. This field is the future ML target.

Example outcome block:

```json
{
  "actual_outcome": "good",
  "comment": "fractured fine and was worth taking"
}
```

Supported `actual_outcome` values:

```text
unknown
good
bad
too_slow
too_unstable
not_enough_power
overheated
wrong_prediction
```

## Current architecture

```text
configs/
  heads.yaml
  modules.yaml
  builds/
    prospector_manual.yaml
    mole_manual.yaml

src/sc_mining/
  domain/
    calculator.py
    config_loader.py
    models.py
  storage/
    event_logger.py
    event_reader.py
  dataset/
    exporter.py
    quality.py
    analytics.py
  ml/
    baseline.py
  ui/
    streamlit_app.py

tests/
  test_calculator.py
  test_event_logger.py
  test_event_reader.py
  test_dataset_exporter.py
  test_dataset_quality.py
  test_dataset_analytics.py
  test_ml_baseline.py
```

## Next planned blocks

1. Collect real labeled events.
2. Collect at least 30-50 real labeled events before trusting the model.
3. Compare formula verdict vs learned prediction in the UI.
4. Add model/version metadata to events.

## Basic analytics

The Saved events tab now includes a Basic analytics block. It uses only labeled rows, where `actual_outcome` is not `unknown`.

It shows:

```text
formula verdict vs actual outcome
good vs not-good feature signals
numeric summaries by actual_outcome
formula diagnostic labels
```

Important diagnostic labels:

```text
correct_take        formula said take and real outcome was good
dangerous_take     formula said take but real outcome was not good
missed_opportunity formula said avoid but real outcome was good
risky_good         formula said risky and real outcome was good
risky_bad          formula said risky and real outcome was not good
correct_avoid      formula said avoid and real outcome was not good
```

This is an analytics step before ML training. It helps decide whether the current rule-based formula is useful and which input features have signal.


## Baseline ML model

Block 9 adds a weak supervised baseline model. It is intentionally simple and should not be treated as final game truth.

Target:

```text
actual_outcome == good -> good
any other labeled outcome -> not_good
```

Model artifacts are written to:

```text
models/mining_outcome_baseline.joblib
reports/baseline_model_report.json
```

Training requires:

```text
actual_outcome != unknown
both good and not-good examples
30+ labeled rows for a weak baseline
```

This is the first MLOps training step: labeled dataset -> model artifact -> evaluation report.
