import json
from pathlib import Path

from sc_mining.ml.real_run import RealMLRunConfig, real_ml_run_result_to_dict, run_real_ml_pipeline


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )


def sample_event(index: int, actual_outcome: str) -> dict:
    is_good = actual_outcome == "good"
    mass = 9000 + index * 3500
    resistance = 0.15 + (index % 4) * 0.08
    instability = 0.03 + (index % 5) * 0.07
    distance = 70 + (index % 6) * 8
    required_power = round((mass / 1000) * (1 + resistance), 3)
    effective_power = round(required_power + (20 if is_good else -8), 3)
    margin = round(effective_power - required_power, 3)
    verdict = "take" if margin > 0 else "need_more_power"

    return {
        "event_id": f"event-{index}",
        "session_id": "real-run-test",
        "timestamp": f"2026-05-25T12:{index:02d}:00+00:00",
        "source": "manual_ui",
        "build": {
            "build_id": "prospector_helix_rieger_focus_v1",
            "ship_type": "prospector",
        },
        "rock": {
            "mass": mass,
            "resistance": resistance,
            "instability": instability,
            "distance": distance,
        },
        "beams": [
            {
                "slot": "main",
                "power_percent": 65,
                "active_modules": [],
            }
        ],
        "result": {
            "required_power": required_power,
            "effective_power": effective_power,
            "margin": margin,
            "risk_score": round(instability, 3),
            "verdict": verdict,
        },
        "outcome": {
            "actual_outcome": actual_outcome,
            "comment": "real run test label",
        },
    }


def test_real_ml_run_reports_not_ready_for_empty_events(tmp_path):
    result = run_real_ml_pipeline(
        RealMLRunConfig(
            events_path=tmp_path / "missing.jsonl",
            dataset_path=tmp_path / "datasets" / "mining_events.csv",
            model_path=tmp_path / "models" / "manual.joblib",
            report_path=tmp_path / "reports" / "manual.json",
            training_runs_path=tmp_path / "reports" / "runs.jsonl",
            active_model_path=tmp_path / "models" / "active_model.json",
            min_labeled_rows=4,
        )
    )

    assert result.status == "not_ready"
    assert result.exported_rows == 0
    assert result.trained is False
    assert result.promoted is False
    assert Path(result.dataset_path).exists()


def test_real_ml_run_trains_manual_model_and_logs_run(tmp_path):
    events_path = tmp_path / "sessions" / "manual_events.jsonl"
    records = [sample_event(i, "good" if i % 2 == 0 else "bad") for i in range(12)]
    write_jsonl(events_path, records)

    result = run_real_ml_pipeline(
        RealMLRunConfig(
            events_path=events_path,
            dataset_path=tmp_path / "datasets" / "mining_events.csv",
            model_path=tmp_path / "models" / "manual.joblib",
            report_path=tmp_path / "reports" / "manual.json",
            training_runs_path=tmp_path / "reports" / "runs.jsonl",
            active_model_path=tmp_path / "models" / "active_model.json",
            min_labeled_rows=6,
            min_test_rows=1,
            min_accuracy=0.0,
            train_if_ready=True,
            promote_if_passed=False,
        )
    )

    assert result.trained is True
    assert result.promoted is False
    assert result.labeled_rows == 12
    assert Path(result.dataset_path).exists()
    assert Path(result.model_path).exists()
    assert Path(result.report_path).exists()
    assert (tmp_path / "reports" / "runs.jsonl").exists()
    assert result.training_run_id
    assert result.training_result["model_source"] == "manual_real_data"


def test_real_ml_run_can_promote_when_gate_passes(tmp_path):
    events_path = tmp_path / "sessions" / "manual_events.jsonl"
    records = [sample_event(i, "good" if i % 2 == 0 else "bad") for i in range(12)]
    write_jsonl(events_path, records)

    result = run_real_ml_pipeline(
        RealMLRunConfig(
            events_path=events_path,
            dataset_path=tmp_path / "datasets" / "mining_events.csv",
            model_path=tmp_path / "models" / "mining_outcome_baseline_manual.joblib",
            report_path=tmp_path / "reports" / "baseline_model_report_manual.json",
            training_runs_path=tmp_path / "reports" / "runs.jsonl",
            active_model_path=tmp_path / "models" / "active_model.json",
            min_labeled_rows=6,
            min_test_rows=1,
            min_accuracy=0.0,
            train_if_ready=True,
            promote_if_passed=True,
        )
    )

    assert result.trained is True
    assert result.can_promote is True
    assert result.promoted is True
    assert result.status == "promoted"
    assert (tmp_path / "models" / "active_model.json").exists()


def test_real_ml_run_result_is_serializable(tmp_path):
    result = run_real_ml_pipeline(
        RealMLRunConfig(
            events_path=tmp_path / "missing.jsonl",
            dataset_path=tmp_path / "datasets" / "mining_events.csv",
            model_path=tmp_path / "models" / "manual.joblib",
            report_path=tmp_path / "reports" / "manual.json",
            training_runs_path=tmp_path / "reports" / "runs.jsonl",
            active_model_path=tmp_path / "models" / "active_model.json",
            min_labeled_rows=4,
            train_if_ready=False,
        )
    )

    payload = real_ml_run_result_to_dict(result)
    assert payload["status"] == "not_ready"
    assert payload["trained"] is False
