from __future__ import annotations

import pandas as pd

from sc_mining.storage.event_reader import LABELED_OUTCOME_VALUES
from sc_mining.ml.prediction_logging import build_prediction_log_dataframe


PREDICTION_EVALUATION_COLUMNS = [
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
    "actual_target",
    "prediction_correct",
    "error_type",
    "outcome_comment",
]


SUMMARY_EMPTY = {
    "event_count": 0,
    "logged_prediction_count": 0,
    "evaluable_prediction_count": 0,
    "correct_count": 0,
    "incorrect_count": 0,
    "accuracy": None,
    "false_good_count": 0,
    "false_not_good_count": 0,
    "model_source_distribution": {},
    "error_type_distribution": {},
}


def outcome_to_binary_target(actual_outcome: str | None) -> str:
    """Map detailed game outcome labels to the baseline binary target."""

    if actual_outcome == "good":
        return "good"
    if actual_outcome in LABELED_OUTCOME_VALUES:
        return "not_good"
    return "unknown"


def classify_prediction_error(ml_prediction: str, actual_target: str) -> str:
    """Return a compact label describing correct predictions and key failure modes."""

    if actual_target == "unknown" or not ml_prediction:
        return "not_evaluable"

    if ml_prediction == actual_target:
        return "correct"

    if ml_prediction == "good" and actual_target == "not_good":
        return "false_good"

    if ml_prediction == "not_good" and actual_target == "good":
        return "false_not_good"

    return "wrong_other"


def build_prediction_evaluation_dataframe(events: pd.DataFrame) -> pd.DataFrame:
    """Evaluate saved ML prediction snapshots once actual outcomes are labeled."""

    if events.empty:
        return pd.DataFrame(columns=PREDICTION_EVALUATION_COLUMNS)

    prediction_log = build_prediction_log_dataframe(events)
    if prediction_log.empty:
        return pd.DataFrame(columns=PREDICTION_EVALUATION_COLUMNS)

    evaluated = prediction_log.copy()
    evaluated["actual_outcome"] = evaluated["actual_outcome"].fillna("unknown").astype(str)
    evaluated["ml_prediction"] = evaluated["ml_prediction"].fillna("").astype(str)
    evaluated["actual_target"] = evaluated["actual_outcome"].map(outcome_to_binary_target)
    evaluated = evaluated[evaluated["actual_target"] != "unknown"].copy()

    if evaluated.empty:
        return pd.DataFrame(columns=PREDICTION_EVALUATION_COLUMNS)

    evaluated["error_type"] = evaluated.apply(
        lambda row: classify_prediction_error(
            ml_prediction=str(row.get("ml_prediction", "")),
            actual_target=str(row.get("actual_target", "unknown")),
        ),
        axis=1,
    )
    evaluated["prediction_correct"] = evaluated["error_type"] == "correct"

    for column in PREDICTION_EVALUATION_COLUMNS:
        if column not in evaluated.columns:
            evaluated[column] = None

    return evaluated[PREDICTION_EVALUATION_COLUMNS]


def build_prediction_evaluation_summary(events: pd.DataFrame) -> dict:
    """Summarize historical inference quality based on later actual-outcome labels."""

    if events.empty:
        return SUMMARY_EMPTY.copy()

    prediction_log = build_prediction_log_dataframe(events)
    evaluated = build_prediction_evaluation_dataframe(events)

    logged_count = int(len(prediction_log))
    evaluable_count = int(len(evaluated))

    if evaluated.empty:
        summary = SUMMARY_EMPTY.copy()
        summary["event_count"] = int(len(events))
        summary["logged_prediction_count"] = logged_count
        return summary

    correct_count = int(evaluated["prediction_correct"].sum())
    incorrect_count = int(evaluable_count - correct_count)

    return {
        "event_count": int(len(events)),
        "logged_prediction_count": logged_count,
        "evaluable_prediction_count": evaluable_count,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy": round(correct_count / evaluable_count, 4) if evaluable_count else None,
        "false_good_count": int((evaluated["error_type"] == "false_good").sum()),
        "false_not_good_count": int((evaluated["error_type"] == "false_not_good").sum()),
        "model_source_distribution": evaluated["ml_model_source"].value_counts(dropna=False).to_dict(),
        "error_type_distribution": evaluated["error_type"].value_counts(dropna=False).to_dict(),
    }


def build_prediction_evaluation_matrix(events: pd.DataFrame) -> pd.DataFrame:
    """Return a small confusion-like matrix: actual target x ML prediction."""

    evaluated = build_prediction_evaluation_dataframe(events)
    if evaluated.empty:
        return pd.DataFrame(columns=["actual_target", "ml_prediction", "count"])

    return (
        evaluated.groupby(["actual_target", "ml_prediction"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["actual_target", "ml_prediction"])
    )
