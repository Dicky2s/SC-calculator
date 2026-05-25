import pandas as pd

from sc_mining.ml.prediction_evaluation import (
    build_prediction_evaluation_dataframe,
    build_prediction_evaluation_matrix,
    build_prediction_evaluation_summary,
    classify_prediction_error,
    outcome_to_binary_target,
)


def event_row(
    event_id: str,
    ml_prediction: str,
    actual_outcome: str,
    model_source: str = "manual_real_data",
) -> dict:
    return {
        "event_id": event_id,
        "session_id": "session-1",
        "timestamp": "2026-05-25T12:00:00+00:00",
        "ship_type": "prospector",
        "build_id": "prospector_helix_rieger_focus_v1",
        "verdict": "take",
        "ml_prediction": ml_prediction,
        "ml_good_probability": 0.8 if ml_prediction == "good" else 0.2,
        "ml_confidence_band": "high",
        "ml_agreement_label": "formula_and_ml_take",
        "ml_model_source": model_source,
        "ml_model_version": "baseline_rf_v1",
        "actual_outcome": actual_outcome,
        "outcome_comment": "checked in game",
    }


def test_outcome_to_binary_target_maps_labels():
    assert outcome_to_binary_target("good") == "good"
    assert outcome_to_binary_target("bad") == "not_good"
    assert outcome_to_binary_target("too_unstable") == "not_good"
    assert outcome_to_binary_target("unknown") == "unknown"
    assert outcome_to_binary_target(None) == "unknown"


def test_classify_prediction_error_labels_failure_modes():
    assert classify_prediction_error("good", "good") == "correct"
    assert classify_prediction_error("not_good", "not_good") == "correct"
    assert classify_prediction_error("good", "not_good") == "false_good"
    assert classify_prediction_error("not_good", "good") == "false_not_good"
    assert classify_prediction_error("", "good") == "not_evaluable"


def test_build_prediction_evaluation_dataframe_filters_unknown_outcomes():
    events = pd.DataFrame(
        [
            event_row("event-1", "good", "good"),
            event_row("event-2", "not_good", "bad"),
            event_row("event-3", "good", "unknown"),
        ]
    )

    evaluated = build_prediction_evaluation_dataframe(events)

    assert len(evaluated) == 2
    assert set(evaluated["event_id"]) == {"event-1", "event-2"}
    assert evaluated["prediction_correct"].tolist() == [True, True]
    assert set(evaluated["actual_target"]) == {"good", "not_good"}


def test_build_prediction_evaluation_summary_counts_errors():
    events = pd.DataFrame(
        [
            event_row("event-1", "good", "good"),
            event_row("event-2", "good", "bad"),
            event_row("event-3", "not_good", "good"),
            event_row("event-4", "not_good", "too_slow"),
            event_row("event-5", "good", "unknown"),
        ]
    )

    summary = build_prediction_evaluation_summary(events)

    assert summary["event_count"] == 5
    assert summary["logged_prediction_count"] == 5
    assert summary["evaluable_prediction_count"] == 4
    assert summary["correct_count"] == 2
    assert summary["incorrect_count"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["false_good_count"] == 1
    assert summary["false_not_good_count"] == 1
    assert summary["error_type_distribution"] == {
        "correct": 2,
        "false_good": 1,
        "false_not_good": 1,
    }


def test_build_prediction_evaluation_matrix_counts_pairs():
    events = pd.DataFrame(
        [
            event_row("event-1", "good", "good"),
            event_row("event-2", "good", "bad"),
            event_row("event-3", "not_good", "bad"),
        ]
    )

    matrix = build_prediction_evaluation_matrix(events)

    actual = {
        (row["actual_target"], row["ml_prediction"]): row["count"]
        for _, row in matrix.iterrows()
    }
    assert actual == {
        ("good", "good"): 1,
        ("not_good", "good"): 1,
        ("not_good", "not_good"): 1,
    }


def test_prediction_evaluation_handles_empty_input():
    events = pd.DataFrame()

    evaluated = build_prediction_evaluation_dataframe(events)
    summary = build_prediction_evaluation_summary(events)
    matrix = build_prediction_evaluation_matrix(events)

    assert evaluated.empty
    assert matrix.empty
    assert summary["event_count"] == 0
    assert summary["accuracy"] is None
