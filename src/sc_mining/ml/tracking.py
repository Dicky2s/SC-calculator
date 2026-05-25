from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


TRAINING_RUNS_PATH = Path("reports") / "training_runs.jsonl"


@dataclass(frozen=True)
class TrainingRunRecord:
    run_id: str
    created_at: str
    model_version: str
    model_source: str
    model_path: str
    report_path: str
    rows_total: int
    rows_used: int
    train_rows: int
    test_rows: int
    accuracy: float
    target_distribution: dict[str, int]
    feature_columns: list[str]
    notes: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_training_run_record(
    training_result: Any,
    notes: str = "",
    run_id: str | None = None,
    created_at: str | None = None,
) -> TrainingRunRecord:
    """Convert a training result object into a durable experiment/run record."""

    payload = asdict(training_result) if hasattr(training_result, "__dataclass_fields__") else dict(training_result)

    return TrainingRunRecord(
        run_id=run_id or str(uuid4()),
        created_at=created_at or _now_iso(),
        model_version=str(payload["model_version"]),
        model_source=str(payload["model_source"]),
        model_path=str(payload["model_path"]),
        report_path=str(payload["report_path"]),
        rows_total=int(payload["rows_total"]),
        rows_used=int(payload["rows_used"]),
        train_rows=int(payload["train_rows"]),
        test_rows=int(payload["test_rows"]),
        accuracy=float(payload["accuracy"]),
        target_distribution={
            str(key): int(value)
            for key, value in dict(payload.get("target_distribution", {})).items()
        },
        feature_columns=[str(column) for column in payload.get("feature_columns", [])],
        notes=str(notes or ""),
    )


def append_training_run(
    training_result: Any,
    path: str | Path = TRAINING_RUNS_PATH,
    notes: str = "",
) -> TrainingRunRecord:
    """Append one model training run into a JSONL experiment history."""

    record = build_training_run_record(training_result=training_result, notes=notes)
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    return record


def load_training_runs(path: str | Path = TRAINING_RUNS_PATH) -> pd.DataFrame:
    """Load training run history as a flat dataframe for UI inspection."""

    source_path = Path(path)
    columns = [
        "run_id",
        "created_at",
        "model_version",
        "model_source",
        "model_path",
        "report_path",
        "rows_total",
        "rows_used",
        "train_rows",
        "test_rows",
        "accuracy",
        "target_distribution",
        "feature_columns",
        "notes",
    ]

    if not source_path.exists():
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"run_id": "invalid_json", "notes": line})

    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)
    for column in columns:
        if column not in df.columns:
            df[column] = None

    for column in ["rows_total", "rows_used", "train_rows", "test_rows"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    return df[columns]


def summarize_training_runs(runs: pd.DataFrame) -> dict[str, Any]:
    if runs.empty:
        return {
            "run_count": 0,
            "model_source_count": 0,
            "best_accuracy": None,
            "latest_run_id": None,
            "latest_model_source": None,
        }

    sorted_runs = runs.sort_values("created_at", ascending=False, na_position="last")
    latest = sorted_runs.iloc[0]

    best_accuracy = runs["accuracy"].max(skipna=True)
    if pd.isna(best_accuracy):
        best_accuracy_value = None
    else:
        best_accuracy_value = round(float(best_accuracy), 4)

    return {
        "run_count": int(len(runs)),
        "model_source_count": int(runs["model_source"].dropna().nunique()),
        "best_accuracy": best_accuracy_value,
        "latest_run_id": str(latest.get("run_id")),
        "latest_model_source": str(latest.get("model_source")),
    }


def latest_training_run_by_source(
    runs: pd.DataFrame,
    model_source: str,
) -> dict[str, Any] | None:
    if runs.empty or "model_source" not in runs.columns:
        return None

    matching = runs[runs["model_source"].astype(str) == str(model_source)].copy()
    if matching.empty:
        return None

    latest = matching.sort_values("created_at", ascending=False, na_position="last").iloc[0]
    return latest.to_dict()
