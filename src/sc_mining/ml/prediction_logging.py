from __future__ import annotations

import pandas as pd


PREDICTION_LOG_COLUMNS = [
    "event_id",
    "session_id",
    "timestamp",
    "ship_type",
    "build_id",
    "verdict",
    "ml_prediction",
    "ml_good_probability",
    "ml_confidence_band",
    "ml_agreement_label",
    "ml_model_source",
    "ml_model_version",
    "actual_outcome",
    "outcome_comment",
]


def has_logged_prediction(row: pd.Series) -> bool:
    prediction = str(row.get("ml_prediction", "") or "").strip()
    return bool(prediction)


def build_prediction_log_dataframe(events: pd.DataFrame) -> pd.DataFrame:
    """Return saved events that contain a formula-vs-ML prediction snapshot."""

    if events.empty:
        return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)

    output = events.copy()
    if "ml_prediction" not in output.columns:
        return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)

    mask = output.apply(has_logged_prediction, axis=1)
    output = output[mask].copy()

    for column in PREDICTION_LOG_COLUMNS:
        if column not in output.columns:
            output[column] = None

    return output[PREDICTION_LOG_COLUMNS]


def build_prediction_log_summary(events: pd.DataFrame) -> dict:
    """Summarize inference snapshots stored inside the event log."""

    if events.empty:
        return {
            "event_count": 0,
            "prediction_count": 0,
            "prediction_coverage_ratio": 0.0,
            "model_source_distribution": {},
            "agreement_distribution": {},
        }

    prediction_log = build_prediction_log_dataframe(events)
    prediction_count = int(len(prediction_log))
    event_count = int(len(events))

    return {
        "event_count": event_count,
        "prediction_count": prediction_count,
        "prediction_coverage_ratio": round(prediction_count / event_count, 4) if event_count else 0.0,
        "model_source_distribution": prediction_log["ml_model_source"].value_counts(dropna=False).to_dict()
        if not prediction_log.empty
        else {},
        "agreement_distribution": prediction_log["ml_agreement_label"].value_counts(dropna=False).to_dict()
        if not prediction_log.empty
        else {},
    }
