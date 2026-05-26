from __future__ import annotations

from dataclasses import dataclass

from sc_mining.domain.models import PowerDistanceObservation
from sc_mining.domain.recommendations import PowerDistanceCandidate


@dataclass(frozen=True)
class FormulaIssueRow:
    source: str
    phase: str
    distance: float | None
    power_percent: float | None
    verdict_or_observation: str
    required_power: float | None = None
    effective_power: float | None = None
    margin: float | None = None
    risk_score: float | None = None
    comment: str = ""


def formula_candidate_to_report_row(
    candidate: PowerDistanceCandidate | None,
    phase: str,
) -> FormulaIssueRow:
    if candidate is None:
        return FormulaIssueRow(
            source="formula",
            phase=phase,
            distance=None,
            power_percent=None,
            verdict_or_observation="not_found",
            comment="Formula/helper did not find a candidate.",
        )

    return FormulaIssueRow(
        source="formula",
        phase=phase,
        distance=round(float(candidate.distance), 3),
        power_percent=round(float(candidate.power_percent), 3),
        verdict_or_observation=str(candidate.verdict),
        required_power=round(float(candidate.required_power), 3),
        effective_power=round(float(candidate.effective_power), 3),
        margin=round(float(candidate.margin), 3),
        risk_score=round(float(candidate.risk_score), 3),
        comment="Formula/helper recommendation.",
    )


def actual_observation_to_report_row(
    distance: float | None,
    power_percent: float | None,
    phase: str,
) -> FormulaIssueRow:
    observation = "warmup" if phase == "warmup" else "stable_hold"
    comment = "Actual in-game observation."
    if distance is None or power_percent is None:
        return FormulaIssueRow(
            source="actual",
            phase=phase,
            distance=None,
            power_percent=None,
            verdict_or_observation="not_filled",
            comment="Fill after checking in game.",
        )

    return FormulaIssueRow(
        source="actual",
        phase=phase,
        distance=round(float(distance), 3),
        power_percent=round(float(power_percent), 3),
        verdict_or_observation=observation,
        comment=comment,
    )


def formula_issue_report_rows(
    *,
    formula_warmup: PowerDistanceCandidate | None,
    formula_stable: PowerDistanceCandidate | None,
    actual_warmup_distance: float | None,
    actual_warmup_power_percent: float | None,
    actual_stable_distance: float | None,
    actual_stable_power_percent: float | None,
) -> list[dict]:
    """Build a compact 4-row formula-vs-actual report.

    Row order is intentional for the UI:
    1. formula warm-up
    2. formula stable hold
    3. actual warm-up
    4. actual stable hold
    """

    rows = [
        formula_candidate_to_report_row(formula_warmup, phase="warmup"),
        formula_candidate_to_report_row(formula_stable, phase="stable"),
        actual_observation_to_report_row(
            actual_warmup_distance,
            actual_warmup_power_percent,
            phase="warmup",
        ),
        actual_observation_to_report_row(
            actual_stable_distance,
            actual_stable_power_percent,
            phase="stable",
        ),
    ]
    return [row.__dict__ for row in rows]


def formula_candidate_to_observation(
    candidate: PowerDistanceCandidate | None,
    phase: str,
) -> PowerDistanceObservation | None:
    if candidate is None:
        return None

    observation = "warmup" if phase == "warmup" else "stable_hold"
    return PowerDistanceObservation(
        distance=float(candidate.distance),
        power_percent=float(candidate.power_percent),
        observation=observation,
        observation_source="formula",
        observation_phase=phase,
        beam_warmed=True,
        held_stable=(phase == "stable"),
        comment="Formula/helper recommendation captured for formula-issue comparison.",
    )


def actual_power_to_observation(
    *,
    distance: float | None,
    power_percent: float | None,
    phase: str,
) -> PowerDistanceObservation | None:
    if distance is None or power_percent is None:
        return None

    observation = "warmup" if phase == "warmup" else "stable_hold"
    return PowerDistanceObservation(
        distance=float(distance),
        power_percent=float(power_percent),
        observation=observation,
        observation_source="actual",
        observation_phase=phase,
        beam_warmed=True,
        held_stable=(phase == "stable"),
        comment="Actual in-game observation captured for formula-issue comparison.",
    )
