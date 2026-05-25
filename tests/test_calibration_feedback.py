import json

from sc_mining.domain.calculator import calculate
from sc_mining.domain.config_loader import load_build, load_heads, load_modules
from sc_mining.domain.models import (
    BeamState,
    CalculationInput,
    CalibrationFeedback,
    PowerDistanceObservation,
    RockInput,
)
from sc_mining.storage.calibration_labeler import (
    has_calibration_observations,
    update_event_calibration,
)
from sc_mining.storage.event_logger import save_calculation_event
from sc_mining.storage.event_reader import load_events_dataframe
from sc_mining.dataset.exporter import build_dataset


def make_event_file(tmp_path):
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = load_build("configs/builds/prospector_helix_2x_rieger.yaml")
    calc_input = CalculationInput(
        rock=RockInput(mass=20728, resistance=0, instability=0.2, distance=15),
        build=build,
        beams=[BeamState(slot="main", power_percent=20)],
    )
    result = calculate(calc_input, heads=heads, modules=modules)
    path = tmp_path / "events.jsonl"
    event = save_calculation_event(
        path=path,
        session_id="manual_2026_05_26",
        calc_input=calc_input,
        result=result,
    )
    return path, event


def test_save_calculation_event_can_store_power_distance_calibration(tmp_path):
    path, event = make_event_file(tmp_path)

    calibration = CalibrationFeedback(
        formula_issue_flag=True,
        observed_distance=15,
        observed_min_warmup_power_percent=78,
        observed_stable_power_percent=81,
        comment="Formula overestimates low-power warm-up.",
        observations=[
            PowerDistanceObservation(
                distance=15,
                power_percent=20,
                observation="no_warmup",
                beam_warmed=False,
                held_stable=False,
            ),
            PowerDistanceObservation(
                distance=15,
                power_percent=81,
                observation="stable_hold",
                beam_warmed=True,
                held_stable=True,
            ),
        ],
    )

    # Append a second event with calibration so event logging path is covered.
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = load_build("configs/builds/prospector_helix_2x_rieger.yaml")
    calc_input = CalculationInput(
        rock=RockInput(mass=20728, resistance=0, instability=0.2, distance=15),
        build=build,
        beams=[BeamState(slot="main", power_percent=20)],
    )
    result = calculate(calc_input, heads=heads, modules=modules)
    saved = save_calculation_event(
        path=path,
        session_id="manual_2026_05_26",
        calc_input=calc_input,
        result=result,
        calibration=calibration,
    )

    assert saved["calibration"]["formula_issue_flag"] is True
    assert saved["calibration"]["observed_min_warmup_power_percent"] == 78
    assert len(saved["calibration"]["observations"]) == 2

    df = load_events_dataframe(path)
    row = df[df["event_id"] == saved["event_id"]].iloc[0]

    assert bool(row["formula_issue_flag"]) is True
    assert row["observed_min_warmup_power_percent"] == 78
    assert row["observed_stable_power_percent"] == 81
    assert row["calibration_attempt_count"] == 2
    assert row["calibration_no_warmup_count"] == 1
    assert row["calibration_stable_hold_count"] == 1

    attempts = json.loads(row["calibration_attempts_json"])
    assert attempts[0]["observation"] == "no_warmup"


def test_update_event_calibration_rewrites_existing_event(tmp_path):
    path, event = make_event_file(tmp_path)

    result = update_event_calibration(
        path=path,
        event_id=event["event_id"],
        calibration=CalibrationFeedback(
            formula_issue_flag=True,
            observed_distance=15,
            observed_min_warmup_power_percent=78,
            observed_stable_power_percent=81,
            observations=[
                PowerDistanceObservation(
                    distance=15,
                    power_percent=20,
                    observation="no_warmup",
                    beam_warmed=False,
                    held_stable=False,
                )
            ],
        ),
    )

    assert result["has_calibration_observations"] is True

    row = load_events_dataframe(path).iloc[0]
    assert row["calibration_attempt_count"] == 1
    assert row["observed_stable_power_percent"] == 81


def test_has_calibration_observations_detects_empty_and_filled_payloads():
    assert has_calibration_observations({}) is False
    assert has_calibration_observations({"formula_issue_flag": True}) is True
    assert has_calibration_observations({"observations": [{"distance": 15, "power_percent": 81}]}) is True


def test_dataset_export_includes_calibration_columns(tmp_path):
    path, event = make_event_file(tmp_path)
    update_event_calibration(
        path=path,
        event_id=event["event_id"],
        calibration=CalibrationFeedback(
            formula_issue_flag=True,
            observed_distance=15,
            observed_min_warmup_power_percent=78,
            observed_stable_power_percent=81,
            comment="stable around 81",
        ),
    )

    dataset = build_dataset(path)
    row = dataset.iloc[0]

    assert bool(row["formula_issue_flag"]) is True
    assert row["observed_min_warmup_power_percent"] == 78
    assert row["observed_stable_power_percent"] == 81
    assert row["calibration_comment"] == "stable around 81"
