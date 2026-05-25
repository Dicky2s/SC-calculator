# SC Mining Assistant

Manual baseline calculator for Star Citizen mining.

Current scope:

- YAML-based mining head/module/build configs
- Prospector and MOLE manual build profiles
- deterministic calculation core
- manual event logging to JSONL
- event dataset reader
- Streamlit UI with:
  - calculator tab
  - saved events tab
  - filters by session, ship, verdict
  - verdict distribution chart
  - numeric dataset summary

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
9 passed
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
  ui/
    streamlit_app.py

tests/
  test_calculator.py
  test_event_logger.py
  test_event_reader.py
```

## Next planned blocks

1. Add labeled outcome field: whether the rock was actually worth taking in-game.
2. Add calibration dataset export.
3. Train a first simple baseline model on saved events.
4. Compare formula verdict vs learned verdict.
