import json

import pandas as pd
import pytest

from sc_mining.dataset.exporter import DATASET_COLUMNS
from sc_mining.ml.baseline import (
    FEATURE_COLUMNS,
    MODEL_VERSION,
    check_training_readiness,
    load_baseline_model,
    predict_good_probability,
    prepare_training_frame,
    train_baseline_model,
)


def make_dataset(rows=40):
    records = []
    for index in range(rows):
        is_good = index % 2 == 0
        records.append(
            {
                "event_id": f"event-{index}",
                "session_id": "session-1",
                "timestamp": "2026-05-25T12:00:00+00:00",
                "source": "test",
                "build_id": "prospector_helix_rieger_focus_v1" if index % 3 else "mole_manual_v1",
                "ship_type": "prospector" if index % 3 else "mole",
                "mass": 12000 + index * 100,
                "resistance": 0.2 + (index % 5) * 0.05,
                "instability": 0.04 + (index % 4) * 0.03,
                "distance": 80 + index % 20,
                "beam_count": 1 if index % 3 else 2,
                "beam_slots": "main" if index % 3 else "main,left",
                "beam_power_sum": 65 + index % 10,
                "required_power": 20 + index % 9,
                "effective_power": 70 + index % 12,
                "margin": 50 if is_good else -10,
                "risk_score": 0.15 if is_good else 0.85,
                "verdict": "take" if is_good else "risky",
                "actual_outcome": "good" if is_good else "too_unstable",
                "is_labeled": True,
                "outcome_comment": "synthetic training row",
            }
        )
    return pd.DataFrame(records, columns=DATASET_COLUMNS)


def test_prepare_training_frame_builds_binary_target():
    dataset = make_dataset(rows=6)

    training_frame = prepare_training_frame(dataset)

    assert list(training_frame.columns) == FEATURE_COLUMNS + ["actual_outcome", "is_good_outcome"]
    assert len(training_frame) == 6
    assert set(training_frame["is_good_outcome"]) == {0, 1}


def test_check_training_readiness_rejects_unlabeled_dataset():
    dataset = make_dataset(rows=6)
    dataset["actual_outcome"] = "unknown"
    dataset["is_labeled"] = False

    readiness = check_training_readiness(dataset, min_labeled_rows=4)

    assert readiness.ready is False
    assert readiness.labeled_rows == 0
    assert "No labeled rows" in readiness.reason


def test_check_training_readiness_rejects_single_binary_class():
    dataset = make_dataset(rows=8)
    dataset["actual_outcome"] = "good"
    dataset["is_labeled"] = True

    readiness = check_training_readiness(dataset, min_labeled_rows=4)

    assert readiness.ready is False
    assert "Only one binary target class" in readiness.reason


def test_train_baseline_model_writes_model_and_report(tmp_path):
    dataset = make_dataset(rows=40)
    model_path = tmp_path / "models" / "baseline.joblib"
    report_path = tmp_path / "reports" / "baseline_report.json"

    result = train_baseline_model(
        dataset=dataset,
        model_path=model_path,
        report_path=report_path,
        min_labeled_rows=10,
        test_size=0.25,
        random_state=7,
        model_source="unit_test",
    )

    assert model_path.exists()
    assert report_path.exists()
    assert result.model_version == MODEL_VERSION
    assert result.model_source == "unit_test"
    assert result.rows_used == 40
    assert result.train_rows == 30
    assert result.test_rows == 10
    assert set(result.target_distribution) == {"0", "1"}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model_version"] == MODEL_VERSION
    assert report["model_source"] == "unit_test"
    assert report["rows_used"] == 40
    assert "classification_report" in report
    assert "confusion_matrix" in report


def test_predict_good_probability_adds_ml_columns(tmp_path):
    dataset = make_dataset(rows=40)
    model_path = tmp_path / "models" / "baseline.joblib"
    report_path = tmp_path / "reports" / "baseline_report.json"

    train_baseline_model(
        dataset=dataset,
        model_path=model_path,
        report_path=report_path,
        min_labeled_rows=10,
        test_size=0.25,
        random_state=7,
    )
    model = load_baseline_model(model_path)

    predictions = predict_good_probability(model, dataset.head(5))

    assert len(predictions) == 5
    assert "ml_good_probability" in predictions.columns
    assert "ml_prediction" in predictions.columns
    assert "ml_model_version" in predictions.columns
    assert predictions["ml_good_probability"].between(0, 1).all()
    assert set(predictions["ml_prediction"]).issubset({"good", "not_good"})


def test_train_baseline_model_raises_when_not_ready(tmp_path):
    dataset = make_dataset(rows=4)

    with pytest.raises(ValueError, match="Collect at least"):
        train_baseline_model(
            dataset=dataset,
            model_path=tmp_path / "model.joblib",
            report_path=tmp_path / "report.json",
            min_labeled_rows=10,
        )
