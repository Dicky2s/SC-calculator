import json

from sc_mining.domain.calibration_report import (
    actual_power_to_observation,
    formula_candidate_to_observation,
    formula_issue_report_rows,
)
from sc_mining.domain.recommendations import PowerDistanceCandidate
from sc_mining.storage.event_reader import flatten_event


def make_candidate(distance: float, power: float, verdict: str = "take") -> PowerDistanceCandidate:
    return PowerDistanceCandidate(
        distance=distance,
        power_percent=power,
        required_power=1000.0,
        effective_power=1120.0,
        margin=120.0,
        margin_ratio=0.12,
        risk_score=0.25,
        verdict=verdict,
    )


def test_formula_issue_report_rows_are_split_formula_then_actual():
    rows = formula_issue_report_rows(
        formula_warmup=make_candidate(15, 88, "risky"),
        formula_stable=make_candidate(16, 100, "take"),
        actual_warmup_distance=15,
        actual_warmup_power_percent=68,
        actual_stable_distance=15,
        actual_stable_power_percent=67,
    )

    assert [row["source"] for row in rows] == ["formula", "formula", "actual", "actual"]
    assert [row["phase"] for row in rows] == ["warmup", "stable", "warmup", "stable"]
    assert rows[0]["power_percent"] == 88
    assert rows[2]["power_percent"] == 68
    assert rows[3]["verdict_or_observation"] == "stable_hold"


def test_formula_rows_do_not_pollute_actual_calibration_counts():
    formula_warmup = formula_candidate_to_observation(make_candidate(15, 88), phase="warmup")
    formula_stable = formula_candidate_to_observation(make_candidate(16, 100), phase="stable")
    actual_warmup = actual_power_to_observation(distance=15, power_percent=68, phase="warmup")
    actual_stable = actual_power_to_observation(distance=15, power_percent=67, phase="stable")

    event = {
        "event_id": "event-1",
        "session_id": "session-1",
        "timestamp": "2026-05-26T11:04:31+00:00",
        "source": "manual_ui",
        "build": {"build_id": "golem", "ship_type": "golem"},
        "rock": {"mass": 15501, "resistance": 0.14, "instability": 0.3135, "distance": 15},
        "beams": [{"slot": "main", "power_percent": 20, "active_modules": []}],
        "result": {"required_power": 3604.884, "effective_power": 826.875, "margin": -2778.009, "risk_score": 0.912, "verdict": "need_more_power"},
        "outcome": {"actual_outcome": "unknown", "comment": ""},
        "resource_yield": {},
        "refinery": {},
        "calibration": {
            "formula_issue_flag": True,
            "observed_min_warmup_power_percent": 68,
            "observed_stable_power_percent": 67,
            "observed_distance": 15,
            "comment": "formula gave 88/100, actual 68/67",
            "observations": [
                formula_warmup.model_dump(),
                formula_stable.model_dump(),
                actual_warmup.model_dump(),
                actual_stable.model_dump(),
            ],
        },
        "labeling": {},
    }

    flat = flatten_event(event)
    attempts = json.loads(flat["calibration_attempts_json"])

    assert flat["calibration_attempt_count"] == 2
    assert flat["calibration_warmup_count"] == 1
    assert flat["calibration_stable_hold_count"] == 1
    assert [row["observation_source"] for row in attempts] == ["formula", "formula", "actual", "actual"]



def test_legacy_four_row_formula_issue_comment_is_inferred_as_formula_then_actual():
    event = {
        "event_id": "legacy-1",
        "session_id": "session-1",
        "timestamp": "2026-05-26T11:04:31+00:00",
        "source": "manual_ui",
        "build": {"build_id": "golem", "ship_type": "golem"},
        "rock": {"mass": 15501, "resistance": 0.14, "instability": 0.3135, "distance": 15},
        "beams": [{"slot": "main", "power_percent": 20, "active_modules": []}],
        "result": {"required_power": 3604.884, "effective_power": 826.875, "margin": -2778.009, "risk_score": 0.912, "verdict": "need_more_power"},
        "outcome": {"actual_outcome": "unknown", "comment": ""},
        "resource_yield": {},
        "refinery": {},
        "calibration": {
            "formula_issue_flag": True,
            "comment": "дало 15 88 и 16 100\nактуальные 15 68 и 15 67",
            "observations": [
                {"distance": 15, "power_percent": 88, "observation": "overpowered", "beam_warmed": True, "held_stable": False, "comment": ""},
                {"distance": 16, "power_percent": 100, "observation": "overpowered", "beam_warmed": True, "held_stable": False, "comment": ""},
                {"distance": 15, "power_percent": 68, "observation": "warmup", "beam_warmed": True, "held_stable": True, "comment": ""},
                {"distance": 15, "power_percent": 67, "observation": "stable_hold", "beam_warmed": True, "held_stable": True, "comment": ""},
            ],
        },
        "labeling": {},
    }

    flat = flatten_event(event)
    attempts = json.loads(flat["calibration_attempts_json"])

    assert flat["calibration_attempt_count"] == 2
    assert [row["observation_source"] for row in attempts] == ["formula", "formula", "actual", "actual"]
    assert [row["observation_phase"] for row in attempts] == ["warmup", "stable", "warmup", "stable"]


def test_actual_stable_observation_becomes_training_power_distance():
    formula_warmup = formula_candidate_to_observation(make_candidate(15, 47), phase="warmup")
    formula_stable = formula_candidate_to_observation(make_candidate(16, 50), phase="stable")
    actual_warmup = actual_power_to_observation(distance=15, power_percent=37, phase="warmup")
    actual_stable = actual_power_to_observation(distance=22, power_percent=48, phase="stable")

    event = {
        "event_id": "event-actual-truth",
        "session_id": "session-1",
        "timestamp": "2026-05-27T12:43:00+00:00",
        "source": "manual_ui",
        "build": {"build_id": "prospector", "ship_type": "prospector"},
        "rock": {"mass": 5245, "resistance": 0.39, "instability": 1.1216, "distance": 15},
        "beams": [{"slot": "main", "power_percent": 20, "active_modules": []}],
        "result": {"required_power": 1719.672, "effective_power": 787.5, "margin": -932.172, "risk_score": 0.3, "verdict": "need_more_power"},
        "outcome": {"actual_outcome": "unknown", "comment": ""},
        "resource_yield": {},
        "refinery": {},
        "calibration": {
            "formula_issue_flag": True,
            "observations": [
                formula_warmup.model_dump(),
                formula_stable.model_dump(),
                actual_warmup.model_dump(),
                actual_stable.model_dump(),
            ],
        },
        "labeling": {},
    }

    flat = flatten_event(event)

    assert flat["scan_distance"] == 15
    assert flat["distance"] == 22
    assert flat["beam_power_sum"] == 48
    assert flat["training_observation_source"] == "actual"
    assert flat["training_observation_phase"] == "stable"
    assert flat["training_observation_distance"] == 22
    assert flat["training_observation_power_percent"] == 48
