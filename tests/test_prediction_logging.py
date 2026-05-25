import pandas as pd

from sc_mining.ml.prediction_logging import (
    PREDICTION_LOG_COLUMNS,
    build_prediction_log_dataframe,
    build_prediction_log_summary,
)


def test_build_prediction_log_dataframe_returns_only_logged_predictions():
    events = pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "session_id": "s1",
                "timestamp": "2026-05-25T12:00:00+00:00",
                "ship_type": "prospector",
                "build_id": "b1",
                "verdict": "take",
                "ml_prediction": "good",
                "ml_good_probability": 0.8,
                "ml_confidence_band": "high_good",
                "ml_agreement_label": "formula_and_ml_take",
                "ml_model_source": "manual_real_data",
                "ml_model_version": "baseline_rf_v1",
                "actual_outcome": "unknown",
                "outcome_comment": "",
            },
            {
                "event_id": "event-2",
                "session_id": "s1",
                "timestamp": "2026-05-25T12:01:00+00:00",
                "ship_type": "prospector",
                "build_id": "b1",
                "verdict": "take",
                "ml_prediction": "",
                "actual_outcome": "unknown",
            },
        ]
    )

    log = build_prediction_log_dataframe(events)

    assert list(log.columns) == PREDICTION_LOG_COLUMNS
    assert len(log) == 1
    assert log.iloc[0]["event_id"] == "event-1"
    assert log.iloc[0]["ml_prediction"] == "good"


def test_build_prediction_log_summary_counts_coverage_and_distributions():
    events = pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "ml_prediction": "good",
                "ml_model_source": "manual_real_data",
                "ml_agreement_label": "formula_and_ml_take",
            },
            {
                "event_id": "event-2",
                "ml_prediction": "not_good",
                "ml_model_source": "synthetic_smoke_test",
                "ml_agreement_label": "ml_warns_against_formula_take",
            },
            {"event_id": "event-3", "ml_prediction": ""},
        ]
    )

    summary = build_prediction_log_summary(events)

    assert summary["event_count"] == 3
    assert summary["prediction_count"] == 2
    assert summary["prediction_coverage_ratio"] == 0.6667
    assert summary["model_source_distribution"] == {
        "manual_real_data": 1,
        "synthetic_smoke_test": 1,
    }
    assert summary["agreement_distribution"] == {
        "formula_and_ml_take": 1,
        "ml_warns_against_formula_take": 1,
    }


def test_prediction_log_handles_empty_input():
    log = build_prediction_log_dataframe(pd.DataFrame())
    summary = build_prediction_log_summary(pd.DataFrame())

    assert log.empty
    assert list(log.columns) == PREDICTION_LOG_COLUMNS
    assert summary["event_count"] == 0
    assert summary["prediction_count"] == 0
