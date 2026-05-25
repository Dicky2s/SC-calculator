from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import random

import pandas as pd

from sc_mining.dataset.exporter import DATASET_COLUMNS, get_dataset_export_summary


NOT_GOOD_OUTCOMES = [
    "bad",
    "too_slow",
    "too_unstable",
    "not_enough_power",
    "overheated",
]


def _round(value: float) -> float:
    return round(float(value), 3)


def _build_required_power(mass: float, resistance: float, distance: float) -> float:
    # Synthetic approximation only. It is intentionally separate from the rule-based calculator
    # so this dataset is used for smoke testing infrastructure, not for real gameplay balance.
    return _round((mass / 1000.0) * (1.0 + resistance * 1.8) * (1.0 + distance / 260.0))


def _risk_score(resistance: float, instability: float, margin: float) -> float:
    shortage_penalty = max(-margin, 0.0) / 80.0
    risk = instability * 2.2 + resistance * 0.18 + shortage_penalty
    return _round(min(max(risk, 0.0), 1.0))


def _verdict(margin: float, risk_score: float) -> str:
    if margin < -5:
        return "need_more_power"
    if risk_score >= 0.72:
        return "risky"
    if margin >= 8 and risk_score <= 0.55:
        return "take"
    return "skip"


def _ship_profile(rng: random.Random) -> tuple[str, str, int, str]:
    if rng.random() < 0.72:
        return "prospector", "prospector_helix_rieger_focus_v1", 1, "main"
    return "mole", "mole_manual_v1", 2, "main,left"


def _good_row(rng: random.Random) -> dict:
    ship_type, build_id, beam_count, beam_slots = _ship_profile(rng)
    mass = rng.uniform(7000, 36000 if ship_type == "prospector" else 62000)
    resistance = rng.uniform(0.05, 0.34)
    instability = rng.uniform(0.01, 0.13)
    distance = rng.uniform(35, 105)
    beam_power_sum = rng.uniform(58, 95) * beam_count
    required_power = _build_required_power(mass, resistance, distance)
    effective_power = required_power + rng.uniform(9, 55)
    margin = _round(effective_power - required_power)
    risk = _risk_score(resistance, instability, margin)

    return {
        "ship_type": ship_type,
        "build_id": build_id,
        "mass": _round(mass),
        "resistance": _round(resistance),
        "instability": _round(instability),
        "distance": _round(distance),
        "beam_count": beam_count,
        "beam_slots": beam_slots,
        "beam_power_sum": _round(beam_power_sum),
        "required_power": required_power,
        "effective_power": _round(effective_power),
        "margin": margin,
        "risk_score": risk,
        "verdict": _verdict(margin, risk),
        "actual_outcome": "good",
    }


def _not_good_row(rng: random.Random) -> dict:
    ship_type, build_id, beam_count, beam_slots = _ship_profile(rng)
    outcome = rng.choice(NOT_GOOD_OUTCOMES)

    if outcome == "not_enough_power":
        mass = rng.uniform(42000, 105000 if ship_type == "prospector" else 155000)
        resistance = rng.uniform(0.38, 0.88)
        instability = rng.uniform(0.04, 0.22)
        distance = rng.uniform(70, 165)
        required_power = _build_required_power(mass, resistance, distance)
        effective_power = max(1.0, required_power - rng.uniform(8, 60))
    elif outcome == "too_unstable":
        mass = rng.uniform(16000, 70000)
        resistance = rng.uniform(0.18, 0.55)
        instability = rng.uniform(0.24, 0.62)
        distance = rng.uniform(45, 140)
        required_power = _build_required_power(mass, resistance, distance)
        effective_power = required_power + rng.uniform(-4, 24)
    elif outcome == "overheated":
        mass = rng.uniform(22000, 84000)
        resistance = rng.uniform(0.30, 0.75)
        instability = rng.uniform(0.18, 0.50)
        distance = rng.uniform(35, 120)
        required_power = _build_required_power(mass, resistance, distance)
        effective_power = required_power + rng.uniform(5, 42)
    elif outcome == "too_slow":
        mass = rng.uniform(36000, 120000)
        resistance = rng.uniform(0.22, 0.70)
        instability = rng.uniform(0.04, 0.26)
        distance = rng.uniform(95, 180)
        required_power = _build_required_power(mass, resistance, distance)
        effective_power = required_power + rng.uniform(-2, 16)
    else:  # bad
        mass = rng.uniform(18000, 90000)
        resistance = rng.uniform(0.22, 0.70)
        instability = rng.uniform(0.08, 0.38)
        distance = rng.uniform(55, 165)
        required_power = _build_required_power(mass, resistance, distance)
        effective_power = required_power + rng.uniform(-25, 20)

    beam_power_sum = rng.uniform(45, 100) * beam_count
    margin = _round(effective_power - required_power)
    risk = _risk_score(resistance, instability, margin)

    return {
        "ship_type": ship_type,
        "build_id": build_id,
        "mass": _round(mass),
        "resistance": _round(resistance),
        "instability": _round(instability),
        "distance": _round(distance),
        "beam_count": beam_count,
        "beam_slots": beam_slots,
        "beam_power_sum": _round(beam_power_sum),
        "required_power": required_power,
        "effective_power": _round(effective_power),
        "margin": margin,
        "risk_score": risk,
        "verdict": _verdict(margin, risk),
        "actual_outcome": outcome,
    }


def generate_synthetic_dataset(
    row_count: int = 100,
    seed: int = 42,
    good_ratio: float = 0.5,
) -> pd.DataFrame:
    """Generate a deterministic synthetic dataset for pipeline smoke tests.

    These rows are not gameplay observations. They are only meant to validate
    dataset quality checks, analytics views, and baseline model training.
    """

    if row_count < 2:
        raise ValueError("row_count must be at least 2 so both target classes can be present")
    if not 0.05 <= good_ratio <= 0.95:
        raise ValueError("good_ratio must be between 0.05 and 0.95")

    rng = random.Random(seed)
    base_timestamp = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    records: list[dict] = []

    good_rows = max(1, min(row_count - 1, round(row_count * good_ratio)))
    labels = ["good"] * good_rows + ["not_good"] * (row_count - good_rows)
    rng.shuffle(labels)

    for index, label in enumerate(labels):
        row = _good_row(rng) if label == "good" else _not_good_row(rng)
        row.update(
            {
                "event_id": f"synthetic-{seed}-{index:05d}",
                "session_id": f"synthetic_seed_{seed}",
                "timestamp": (base_timestamp + timedelta(minutes=index)).isoformat(),
                "source": "synthetic",
                "is_labeled": True,
                "outcome_comment": "synthetic smoke-test row; do not use as real gameplay data",
            }
        )
        records.append(row)

    return pd.DataFrame(records, columns=DATASET_COLUMNS)


def export_synthetic_dataset(
    output_path: str | Path,
    row_count: int = 100,
    seed: int = 42,
    good_ratio: float = 0.5,
) -> pd.DataFrame:
    dataset = generate_synthetic_dataset(
        row_count=row_count,
        seed=seed,
        good_ratio=good_ratio,
    )

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(target_path, index=False, encoding="utf-8")
    return dataset


def synthetic_summary(dataset: pd.DataFrame) -> dict:
    summary = get_dataset_export_summary(dataset)
    summary["source"] = "synthetic"
    summary["warning"] = "Synthetic rows are for smoke tests only, not for real model quality."
    return summary
