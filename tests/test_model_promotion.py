import json

from sc_mining.ml.promotion import (
    PromotionCriteria,
    evaluate_model_promotion,
    load_training_report,
    promotion_decision_to_dict,
)
from sc_mining.ml.registry import MODEL_SOURCE_MANUAL_REAL, MODEL_SOURCE_SYNTHETIC


def write_report(path, **overrides):
    payload = {
        "model_version": "baseline_rf_v1",
        "model_source": MODEL_SOURCE_MANUAL_REAL,
        "rows_used": 50,
        "test_rows": 15,
        "accuracy": 0.75,
        "target_distribution": {"0": 25, "1": 25},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_load_training_report_reads_json(tmp_path):
    report_path = tmp_path / "report.json"
    payload = write_report(report_path)

    assert load_training_report(report_path) == payload


def test_promotion_passes_for_manual_model_with_enough_quality(tmp_path):
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"
    model_path.write_bytes(b"fake-model")
    write_report(report_path)

    decision = evaluate_model_promotion(model_path, report_path)

    assert decision.status == "pass"
    assert decision.can_promote is True
    assert decision.reasons == []
    assert decision.metrics["model_source"] == MODEL_SOURCE_MANUAL_REAL


def test_promotion_fails_when_model_is_missing(tmp_path):
    report_path = tmp_path / "report.json"
    write_report(report_path)

    decision = evaluate_model_promotion(tmp_path / "missing.joblib", report_path)

    assert decision.status == "fail"
    assert decision.can_promote is False
    assert any("Model artifact is missing" in reason for reason in decision.reasons)


def test_promotion_fails_for_synthetic_source(tmp_path):
    model_path = tmp_path / "synthetic.joblib"
    report_path = tmp_path / "report.json"
    model_path.write_bytes(b"fake-model")
    write_report(report_path, model_source=MODEL_SOURCE_SYNTHETIC)

    decision = evaluate_model_promotion(model_path, report_path)

    assert decision.status == "fail"
    assert decision.can_promote is False
    assert any("expected manual_real_data" in reason for reason in decision.reasons)


def test_promotion_fails_when_rows_are_too_low(tmp_path):
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"
    model_path.write_bytes(b"fake-model")
    write_report(report_path, rows_used=12)

    decision = evaluate_model_promotion(model_path, report_path)

    assert decision.status == "fail"
    assert any("Only 12 rows" in reason for reason in decision.reasons)


def test_promotion_fails_for_one_binary_class(tmp_path):
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"
    model_path.write_bytes(b"fake-model")
    write_report(report_path, target_distribution={"1": 50})

    decision = evaluate_model_promotion(model_path, report_path)

    assert decision.status == "fail"
    assert any("fewer than two binary classes" in reason for reason in decision.reasons)


def test_promotion_can_warn_on_prediction_evaluation(tmp_path):
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"
    model_path.write_bytes(b"fake-model")
    write_report(report_path)

    decision = evaluate_model_promotion(
        model_path,
        report_path,
        criteria=PromotionCriteria(max_false_good_rate=0.10),
        prediction_evaluation_summary={
            "evaluable_prediction_count": 20,
            "false_good_count": 5,
        },
    )

    assert decision.status == "warn"
    assert decision.can_promote is True
    assert any("False-good rate" in warning for warning in decision.warnings)


def test_promotion_decision_to_dict_is_serializable(tmp_path):
    report_path = tmp_path / "report.json"
    decision = evaluate_model_promotion(tmp_path / "missing.joblib", report_path)
    payload = promotion_decision_to_dict(decision)

    assert payload["status"] == "fail"
    assert "metrics" in payload
