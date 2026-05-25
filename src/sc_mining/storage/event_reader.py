from pathlib import Path
import json

import pandas as pd


EVENT_COLUMNS = [
    "event_id",
    "session_id",
    "timestamp",
    "source",
    "build_id",
    "ship_type",
    "mass",
    "resistance",
    "instability",
    "distance",
    "beam_count",
    "beam_slots",
    "beam_power_sum",
    "required_power",
    "effective_power",
    "margin",
    "risk_score",
    "verdict",
]


def read_jsonl(path: str | Path) -> list[dict]:
    input_path = Path(path)

    if not input_path.exists():
        return []

    records: list[dict] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {input_path}, line {line_number}: {error}"
                ) from error

    return records


def flatten_event(event: dict) -> dict:
    build = event.get("build", {})
    rock = event.get("rock", {})
    result = event.get("result", {})
    beams = event.get("beams", [])

    beam_slots = [beam.get("slot", "") for beam in beams]
    beam_power_sum = sum(float(beam.get("power_percent", 0.0)) for beam in beams)

    return {
        "event_id": event.get("event_id"),
        "session_id": event.get("session_id"),
        "timestamp": event.get("timestamp"),
        "source": event.get("source"),
        "build_id": build.get("build_id"),
        "ship_type": build.get("ship_type"),
        "mass": rock.get("mass"),
        "resistance": rock.get("resistance"),
        "instability": rock.get("instability"),
        "distance": rock.get("distance"),
        "beam_count": len(beams),
        "beam_slots": ", ".join(beam_slots),
        "beam_power_sum": beam_power_sum,
        "required_power": result.get("required_power"),
        "effective_power": result.get("effective_power"),
        "margin": result.get("margin"),
        "risk_score": result.get("risk_score"),
        "verdict": result.get("verdict"),
    }


def load_events_dataframe(path: str | Path) -> pd.DataFrame:
    records = read_jsonl(path)

    if not records:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    rows = [flatten_event(record) for record in records]
    df = pd.DataFrame(rows)

    for column in EVENT_COLUMNS:
        if column not in df.columns:
            df[column] = None

    return df[EVENT_COLUMNS]


def get_events_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "event_count": 0,
            "session_count": 0,
            "build_count": 0,
            "ship_count": 0,
        }

    return {
        "event_count": int(len(df)),
        "session_count": int(df["session_id"].nunique()),
        "build_count": int(df["build_id"].nunique()),
        "ship_count": int(df["ship_type"].nunique()),
    }
