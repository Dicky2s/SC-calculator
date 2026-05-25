from pathlib import Path
import json

import pandas as pd


EVENT_COLUMNS = [
    "event_id",
    "session_id",
    "timestamp",
    "source",
    "operator_name",
    "crew_size",
    "run_tag",
    "build_id",
    "ship_type",
    "mass",
    "resistance",
    "instability",
    "distance",
    "beam_count",
    "beam_slots",
    "beam_power_sum",
    "required_power",
    "effective_power",
    "margin",
    "risk_score",
    "verdict",
    "ml_model_available",
    "ml_model_version",
    "ml_model_path",
    "ml_model_source",
    "ml_formula_expected_outcome",
    "ml_prediction",
    "ml_good_probability",
    "ml_confidence_band",
    "ml_agreement_label",
    "ml_prediction_captured_at",
    "actual_outcome",
    "outcome_comment",
    "primary_resource",
    "resource_percent",
    "raw_scu_estimate",
    "total_scu_estimate",
    "refined_scu_estimate",
    "estimated_value_auec",
    "mining_time_seconds",
    "resource_comment",
    "resource_count",
    "resource_names",
    "total_resource_percent",
    "resources_json",
    "refinery_method",
    "refinery_location",
    "refinery_start_at",
    "refinery_complete_at",
    "refined_scu_actual",
    "refined_value_auec",
    "refinery_fee_auec",
    "sell_value_auec",
    "refinery_comment",
    "refined_resource_count",
    "refined_resource_names",
    "total_refined_scu_actual",
    "total_resource_sell_value_auec",
    "refined_resources_json",
    "formula_issue_flag",
    "observed_min_warmup_power_percent",
    "observed_stable_power_percent",
    "observed_distance",
    "calibration_comment",
    "calibration_attempt_count",
    "calibration_no_warmup_count",
    "calibration_warmup_count",
    "calibration_stable_hold_count",
    "calibration_attempts_json",
    "label_source",
    "labeled_at",
]


LABELED_OUTCOME_VALUES = {
    "good",
    "bad",
    "too_slow",
    "too_unstable",
    "not_enough_power",
    "overheated",
    "wrong_prediction",
}


def read_jsonl(path: str | Path) -> list[dict]:
    input_path = Path(path)

    if not input_path.exists():
        return []

    records: list[dict] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {input_path}, line {line_number}: {error}"
                ) from error

    return records


def _valid_resource_rows(resource_yield: dict) -> list[dict]:
    resources = resource_yield.get("resources") or []
    valid: list[dict] = []

    for row in resources:
        name = str(row.get("resource_name", "unknown") or "unknown")
        percent = row.get("resource_percent")
        raw_scu = row.get("raw_scu_estimate")
        comment = row.get("comment", "")

        if name == "unknown" and percent in (None, "", 0, 0.0) and raw_scu in (None, "", 0, 0.0):
            continue

        valid.append(
            {
                "resource_name": name,
                "resource_percent": percent,
                "raw_scu_estimate": raw_scu,
                "comment": comment,
            }
        )

    # Backward compatibility with old single-resource events.
    if not valid:
        primary_resource = str(resource_yield.get("primary_resource", "unknown") or "unknown")
        resource_percent = resource_yield.get("resource_percent")
        raw_scu_estimate = resource_yield.get("raw_scu_estimate")
        if primary_resource != "unknown" or resource_percent is not None or raw_scu_estimate is not None:
            valid.append(
                {
                    "resource_name": primary_resource,
                    "resource_percent": resource_percent,
                    "raw_scu_estimate": raw_scu_estimate,
                    "comment": resource_yield.get("comment", ""),
                }
            )

    return valid


def _sum_percent(resources: list[dict]) -> float | None:
    total = 0.0
    found = False
    for row in resources:
        value = row.get("resource_percent")
        if value in (None, ""):
            continue
        total += float(value)
        found = True
    return round(total, 3) if found else None


def _valid_refined_resource_rows(refinery: dict) -> list[dict]:
    refined_resources = refinery.get("refined_resources") or []
    valid: list[dict] = []

    for row in refined_resources:
        name = str(row.get("resource_name", "unknown") or "unknown")
        refined_scu = row.get("refined_scu_actual")
        sell_value = row.get("sell_value_auec")
        comment = row.get("comment", "")

        if name == "unknown" and refined_scu in (None, "", 0, 0.0) and sell_value in (None, "", 0, 0.0):
            continue

        valid.append(
            {
                "resource_name": name,
                "refined_scu_actual": refined_scu,
                "sell_value_auec": sell_value,
                "comment": comment,
            }
        )

    return valid


def _sum_numeric(rows: list[dict], key: str) -> float | None:
    total = 0.0
    found = False
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        total += float(value)
        found = True
    return round(total, 3) if found else None



def _valid_calibration_observation_rows(calibration: dict) -> list[dict]:
    observations = calibration.get("observations") or []
    valid: list[dict] = []

    for row in observations:
        distance = row.get("distance")
        power_percent = row.get("power_percent")
        observation = str(row.get("observation", "unknown") or "unknown")
        beam_warmed = row.get("beam_warmed")
        held_stable = row.get("held_stable")
        comment = str(row.get("comment", "") or "")

        if distance in (None, "") or power_percent in (None, ""):
            if observation == "unknown" and not comment.strip():
                continue

        valid.append(
            {
                "distance": distance,
                "power_percent": power_percent,
                "observation": observation,
                "beam_warmed": beam_warmed,
                "held_stable": held_stable,
                "comment": comment,
            }
        )

    return valid


def _count_observations(rows: list[dict], observation: str) -> int:
    return sum(1 for row in rows if row.get("observation") == observation)

def flatten_event(event: dict) -> dict:
    build = event.get("build", {})
    run_context = event.get("run_context", {})
    rock = event.get("rock", {})
    result = event.get("result", {})
    outcome = event.get("outcome", {})
    resource_yield = event.get("resource_yield", {})
    refinery = event.get("refinery", {})
    calibration = event.get("calibration", {})
    ml_prediction = event.get("ml_prediction", {})
    labeling = event.get("labeling", {})
    beams = event.get("beams", [])

    beam_slots = [beam.get("slot", "") for beam in beams]
    beam_power_sum = sum(float(beam.get("power_percent", 0.0)) for beam in beams)

    resources = _valid_resource_rows(resource_yield)
    resource_names = [row.get("resource_name", "unknown") for row in resources]
    refined_resources = _valid_refined_resource_rows(refinery)
    refined_resource_names = [row.get("resource_name", "unknown") for row in refined_resources]
    calibration_observations = _valid_calibration_observation_rows(calibration)

    return {
        "event_id": event.get("event_id"),
        "session_id": event.get("session_id"),
        "timestamp": event.get("timestamp"),
        "source": event.get("source"),
        "operator_name": run_context.get("operator_name", ""),
        "crew_size": run_context.get("crew_size", 1),
        "run_tag": run_context.get("run_tag", ""),
        "build_id": build.get("build_id"),
        "ship_type": build.get("ship_type"),
        "mass": rock.get("mass"),
        "resistance": rock.get("resistance"),
        "instability": rock.get("instability"),
        "distance": rock.get("distance"),
        "beam_count": len(beams),
        "beam_slots": ", ".join(beam_slots),
        "beam_power_sum": beam_power_sum,
        "required_power": result.get("required_power"),
        "effective_power": result.get("effective_power"),
        "margin": result.get("margin"),
        "risk_score": result.get("risk_score"),
        "verdict": result.get("verdict"),
        "ml_model_available": bool(ml_prediction.get("model_available", False)),
        "ml_model_version": ml_prediction.get("model_version", ""),
        "ml_model_path": ml_prediction.get("model_path", ""),
        "ml_model_source": ml_prediction.get("model_source", ""),
        "ml_formula_expected_outcome": ml_prediction.get("formula_expected_outcome", ""),
        "ml_prediction": ml_prediction.get("prediction", ""),
        "ml_good_probability": ml_prediction.get("good_probability"),
        "ml_confidence_band": ml_prediction.get("confidence_band", ""),
        "ml_agreement_label": ml_prediction.get("agreement_label", ""),
        "ml_prediction_captured_at": ml_prediction.get("captured_at", ""),
        "actual_outcome": outcome.get("actual_outcome", "unknown"),
        "outcome_comment": outcome.get("comment", ""),
        "primary_resource": resource_yield.get("primary_resource", "unknown"),
        "resource_percent": resource_yield.get("resource_percent"),
        "raw_scu_estimate": resource_yield.get("raw_scu_estimate"),
        "total_scu_estimate": resource_yield.get("total_scu_estimate"),
        "refined_scu_estimate": resource_yield.get("refined_scu_estimate"),
        "estimated_value_auec": resource_yield.get("estimated_value_auec"),
        "mining_time_seconds": resource_yield.get("mining_time_seconds"),
        "resource_comment": resource_yield.get("comment", ""),
        "resource_count": len(resources),
        "resource_names": ", ".join(resource_names),
        "total_resource_percent": _sum_percent(resources),
        "resources_json": json.dumps(resources, ensure_ascii=False),
        "refinery_method": refinery.get("refinery_method", "unknown"),
        "refinery_location": refinery.get("refinery_location", ""),
        "refinery_start_at": refinery.get("refinery_start_at", ""),
        "refinery_complete_at": refinery.get("refinery_complete_at", ""),
        "refined_scu_actual": refinery.get("refined_scu_actual"),
        "refined_value_auec": refinery.get("refined_value_auec"),
        "refinery_fee_auec": refinery.get("refinery_fee_auec"),
        "sell_value_auec": refinery.get("sell_value_auec"),
        "refinery_comment": refinery.get("comment", ""),
        "refined_resource_count": len(refined_resources),
        "refined_resource_names": ", ".join(refined_resource_names),
        "total_refined_scu_actual": _sum_numeric(refined_resources, "refined_scu_actual") or refinery.get("refined_scu_actual"),
        "total_resource_sell_value_auec": _sum_numeric(refined_resources, "sell_value_auec") or refinery.get("sell_value_auec"),
        "refined_resources_json": json.dumps(refined_resources, ensure_ascii=False),
        "formula_issue_flag": bool(calibration.get("formula_issue_flag", False)),
        "observed_min_warmup_power_percent": calibration.get("observed_min_warmup_power_percent"),
        "observed_stable_power_percent": calibration.get("observed_stable_power_percent"),
        "observed_distance": calibration.get("observed_distance"),
        "calibration_comment": calibration.get("comment", ""),
        "calibration_attempt_count": len(calibration_observations),
        "calibration_no_warmup_count": _count_observations(calibration_observations, "no_warmup"),
        "calibration_warmup_count": _count_observations(calibration_observations, "warmup"),
        "calibration_stable_hold_count": _count_observations(calibration_observations, "stable_hold"),
        "calibration_attempts_json": json.dumps(calibration_observations, ensure_ascii=False),
        "label_source": labeling.get("label_source", ""),
        "labeled_at": labeling.get("labeled_at", ""),
    }


def load_events_dataframe(path: str | Path) -> pd.DataFrame:
    records = read_jsonl(path)

    if not records:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    rows = [flatten_event(record) for record in records]
    df = pd.DataFrame(rows)

    for column in EVENT_COLUMNS:
        if column not in df.columns:
            df[column] = None

    return df[EVENT_COLUMNS]


def get_events_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "event_count": 0,
            "labeled_event_count": 0,
            "unlabeled_event_count": 0,
            "session_count": 0,
            "build_count": 0,
            "ship_count": 0,
        }

    actual_outcome = df["actual_outcome"].fillna("unknown")
    labeled_mask = actual_outcome.isin(LABELED_OUTCOME_VALUES)

    return {
        "event_count": int(len(df)),
        "labeled_event_count": int(labeled_mask.sum()),
        "unlabeled_event_count": int((~labeled_mask).sum()),
        "session_count": int(df["session_id"].nunique()),
        "build_count": int(df["build_id"].nunique()),
        "ship_count": int(df["ship_type"].nunique()),
    }
