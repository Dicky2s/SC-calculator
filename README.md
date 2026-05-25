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
24 passed
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
  ui/
    streamlit_app.py

tests/
  test_calculator.py
  test_event_logger.py
  test_event_reader.py
```

## Next planned blocks

1. Collect real labeled events.
2. Train a first simple baseline model on saved events.
3. Add model evaluation report.
4. Compare formula verdict vs learned prediction in the UI.
