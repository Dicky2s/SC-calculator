from __future__ import annotations

from dataclasses import asdict, dataclass

from sc_mining.domain.calculator import calculate, find_head_build, head_max_range
from sc_mining.domain.models import (
    BeamState,
    CalculationInput,
    HeadConfig,
    ModuleConfig,
    RockInput,
)


@dataclass(frozen=True)
class PowerDistanceCandidate:
    distance: float
    power_percent: float
    required_power: float
    effective_power: float
    margin: float
    margin_ratio: float
    risk_score: float
    verdict: str


@dataclass(frozen=True)
class PowerDistanceRecommendation:
    # New semantic split:
    # - minimum_reaction: first point where the rock reacts; may be too slow/stalled.
    # - recommended_warmup: practical charge-up point with extra margin over minimum.
    # - stable_hold: lower/near-threshold point for keeping charge after entering the window.
    # - upper_safe: last comfortable point before overpower/overshoot risk gets high.
    minimum_reaction: PowerDistanceCandidate | None
    recommended_warmup: PowerDistanceCandidate | None
    stable_hold: PowerDistanceCandidate | None
    upper_safe: PowerDistanceCandidate | None
    scanned_count: int
    note: str
    best_available: PowerDistanceCandidate | None
    limiting_reason: str
    required_multiplier: float | None

    @property
    def minimum_warmup(self) -> PowerDistanceCandidate | None:
        """Backward-compatible alias for older UI/tests/datasets.

        Older code called this value ``minimum_warmup``. The new wording is
        ``minimum_reaction`` because it can heat too slowly to be a practical
        warm-up recommendation.
        """

        return self.minimum_reaction


def _candidate_to_dict(candidate: PowerDistanceCandidate | None) -> dict | None:
    return None if candidate is None else asdict(candidate)


def recommendation_to_dict(recommendation: PowerDistanceRecommendation) -> dict:
    minimum_reaction = _candidate_to_dict(recommendation.minimum_reaction)
    return {
        "minimum_reaction": minimum_reaction,
        # Backward-compatible key kept for already-written analysis code.
        "minimum_warmup": minimum_reaction,
        "recommended_warmup": _candidate_to_dict(recommendation.recommended_warmup),
        "stable_hold": _candidate_to_dict(recommendation.stable_hold),
        "upper_safe": _candidate_to_dict(recommendation.upper_safe),
        "best_available": _candidate_to_dict(recommendation.best_available),
        "scanned_count": recommendation.scanned_count,
        "note": recommendation.note,
        "limiting_reason": recommendation.limiting_reason,
        "required_multiplier": recommendation.required_multiplier,
    }


def _with_candidate_power_and_distance(
    calc_input: CalculationInput,
    distance: float,
    power_percent: float,
) -> CalculationInput:
    return CalculationInput(
        rock=RockInput(
            mass=calc_input.rock.mass,
            resistance=calc_input.rock.resistance,
            instability=calc_input.rock.instability,
            distance=float(distance),
        ),
        build=calc_input.build,
        beams=[
            BeamState(
                slot=beam.slot,
                power_percent=float(power_percent),
                active_modules=list(beam.active_modules),
            )
            for beam in calc_input.beams
        ],
    )


def _max_valid_scan_distance(
    calc_input: CalculationInput,
    heads: dict[str, HeadConfig],
    requested_max_distance: int,
) -> int:
    """Cap recommendation scans to the shortest max range among scanned heads."""

    if not calc_input.beams:
        return requested_max_distance

    max_ranges: list[float] = []
    for beam in calc_input.beams:
        head_build = find_head_build(calc_input.build, beam.slot)
        if head_build.head_id in heads:
            max_ranges.append(head_max_range(heads[head_build.head_id]))

    if not max_ranges:
        return requested_max_distance

    return int(min(float(requested_max_distance), min(max_ranges)))


def scan_power_distance_grid(
    calc_input: CalculationInput,
    heads: dict[str, HeadConfig],
    modules: dict[str, ModuleConfig],
    min_distance: int = 10,
    max_distance: int = 120,
    distance_step: int = 1,
    min_power: int = 20,
    max_power: int = 100,
    power_step: int = 1,
) -> list[PowerDistanceCandidate]:
    """Scan possible distance/power pairs using the same calculator formula.

    This is a rule-based guidance helper, not a real game autopilot. It uses the
    current enabled beam slots and active modules, and applies one shared power
    percentage to all enabled beams during the scan.
    """

    if not calc_input.beams:
        return []

    candidates: list[PowerDistanceCandidate] = []
    capped_max_distance = _max_valid_scan_distance(calc_input, heads, max_distance)

    for distance in range(min_distance, capped_max_distance + 1, distance_step):
        for power in range(min_power, max_power + 1, power_step):
            candidate_input = _with_candidate_power_and_distance(
                calc_input=calc_input,
                distance=float(distance),
                power_percent=float(power),
            )
            try:
                result = calculate(candidate_input, heads=heads, modules=modules)
            except ValueError:
                continue
            margin_ratio = result.margin / max(result.required_power, 1.0)
            candidates.append(
                PowerDistanceCandidate(
                    distance=float(distance),
                    power_percent=float(power),
                    required_power=result.required_power,
                    effective_power=result.effective_power,
                    margin=result.margin,
                    margin_ratio=round(margin_ratio, 4),
                    risk_score=result.risk_score,
                    verdict=result.verdict,
                )
            )

    return candidates


def _distance_priority(candidate: PowerDistanceCandidate, current_distance: float) -> tuple[int, float]:
    distance_delta = abs(candidate.distance - current_distance)
    return (0 if distance_delta <= 5 else 1, distance_delta)


def _safe_reaction_candidates(candidates: list[PowerDistanceCandidate]) -> list[PowerDistanceCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.margin >= 0
        and candidate.risk_score < 0.75
        and candidate.verdict not in {"skip", "need_more_power"}
    ]


def choose_minimum_reaction(
    candidates: list[PowerDistanceCandidate],
    current_distance: float,
) -> PowerDistanceCandidate | None:
    """Find the first point where the rock reacts.

    This is intentionally a threshold, not the main practical recommendation.
    In-game this can mean "it starts warming but does not really climb".
    """

    # First reaction is a threshold detector, so it must not require the same
    # low-risk band as a practical warm-up recommendation. High-instability
    # rocks can have a real first reaction at a risk score above 0.75.
    valid = [
        candidate
        for candidate in candidates
        if candidate.margin >= 0
        and candidate.verdict not in {"skip", "need_more_power"}
        and candidate.risk_score < 0.90
    ]
    if not valid:
        return None

    return sorted(
        valid,
        key=lambda candidate: (
            *_distance_priority(candidate, current_distance),
            candidate.power_percent,
            candidate.risk_score,
            abs(candidate.margin_ratio),
        ),
    )[0]


def choose_minimum_warmup(candidates: list[PowerDistanceCandidate]) -> PowerDistanceCandidate | None:
    """Backward-compatible wrapper for older tests/callers.

    Without current-distance context it behaves like the old helper: lowest safe
    power first.
    """

    valid = _safe_reaction_candidates(candidates)
    if not valid:
        return None
    return sorted(
        valid,
        key=lambda candidate: (
            candidate.power_percent,
            candidate.risk_score,
            -candidate.distance,
            abs(candidate.margin_ratio),
        ),
    )[0]


def choose_recommended_warmup(
    candidates: list[PowerDistanceCandidate],
    current_distance: float,
    minimum_reaction: PowerDistanceCandidate | None,
) -> PowerDistanceCandidate | None:
    """Find a conservative charge-up point above the first reaction threshold.

    In-game calibration showed that the old warm-up target was too aggressive:
    the rock reacted, but the suggested value could immediately overheat or
    keep climbing. The helper now adds only a small buffer over first reaction.
    If this still warms without growth, the user should add 1-3% and record the
    actual point; that observation is more valuable than a hard-coded global
    offset.
    """

    if minimum_reaction is None:
        return None

    target_margin_ratio = max(0.035, minimum_reaction.margin_ratio + 0.02)
    minimum_power = min(100.0, minimum_reaction.power_percent + 1.0)
    valid = [
        candidate
        for candidate in _safe_reaction_candidates(candidates)
        if candidate.power_percent >= minimum_power
        and 0.01 <= candidate.margin_ratio <= 0.14
        and candidate.risk_score < 0.72
    ]

    if not valid:
        valid = [
            candidate
            for candidate in _safe_reaction_candidates(candidates)
            if candidate.power_percent >= minimum_power
            and candidate.risk_score < 0.74
        ]

    if not valid:
        return minimum_reaction

    return sorted(
        valid,
        key=lambda candidate: (
            *_distance_priority(candidate, current_distance),
            abs(candidate.margin_ratio - target_margin_ratio),
            candidate.power_percent,
            candidate.risk_score,
        ),
    )[0]


def choose_stable_hold(
    candidates: list[PowerDistanceCandidate],
    current_distance: float,
    recommended_warmup: PowerDistanceCandidate | None = None,
    minimum_reaction: PowerDistanceCandidate | None = None,
) -> PowerDistanceCandidate | None:
    """Find a conservative hold point after the rock is already warming.

    Hold power can be slightly below the cold-start reaction threshold because
    the goal is to maintain charge after entering/approaching the green window,
    not to start from zero. Therefore this selector allows a small negative
    margin band and prefers a value below minimum reaction when available.
    """

    upper_power = recommended_warmup.power_percent if recommended_warmup is not None else 100.0
    if minimum_reaction is not None:
        upper_power = min(upper_power, minimum_reaction.power_percent)

    target_margin_ratio = -0.015
    valid = [
        candidate
        for candidate in candidates
        if candidate.power_percent <= upper_power
        and -0.055 <= candidate.margin_ratio <= 0.025
        and candidate.risk_score < 0.72
        and candidate.verdict != "skip"
    ]

    if not valid and minimum_reaction is not None:
        valid = [
            candidate
            for candidate in candidates
            if candidate.power_percent <= upper_power
            and -0.08 <= candidate.margin_ratio <= max(0.04, minimum_reaction.margin_ratio + 0.005)
            and candidate.risk_score < 0.78
            and candidate.verdict != "skip"
        ]

    if not valid:
        return None

    return sorted(
        valid,
        key=lambda candidate: (
            *_distance_priority(candidate, current_distance),
            abs(candidate.margin_ratio - target_margin_ratio),
            candidate.power_percent,
            candidate.risk_score,
        ),
    )[0]


def choose_upper_safe(
    candidates: list[PowerDistanceCandidate],
    current_distance: float,
) -> PowerDistanceCandidate | None:
    """Find the highest still-comfortable power before likely overshoot."""

    valid = [
        candidate
        for candidate in _safe_reaction_candidates(candidates)
        if candidate.margin_ratio <= 0.45
        and candidate.risk_score < 0.72
    ]
    if not valid:
        return None

    return sorted(
        valid,
        key=lambda candidate: (
            *_distance_priority(candidate, current_distance),
            -candidate.power_percent,
            candidate.risk_score,
        ),
    )[0]


def build_recommendation_scan_input(calc_input: CalculationInput) -> CalculationInput:
    """Create a scan input that ignores any placeholder/current UI power value.

    The recommendation helper finds the power percent itself, so it must not
    depend on a manually selected slider value. It keeps only the enabled beam
    slots and their selected active modules, then lets the grid scan replace
    power from 20% to 100%. If the user disables all beams, the helper returns
    no candidates instead of silently scanning every installed head.
    """

    beams = [
        BeamState(
            slot=beam.slot,
            power_percent=20.0,
            active_modules=list(beam.active_modules),
        )
        for beam in calc_input.beams
    ]

    return CalculationInput(
        rock=RockInput(
            mass=calc_input.rock.mass,
            resistance=calc_input.rock.resistance,
            instability=calc_input.rock.instability,
            distance=calc_input.rock.distance,
        ),
        build=calc_input.build,
        beams=beams,
    )


def select_recommendation_candidate(
    recommendation: PowerDistanceRecommendation,
) -> PowerDistanceCandidate | None:
    """Choose the candidate that should represent the formula snapshot.

    Stable hold is the most useful diagnostic snapshot. If it is missing, use
    practical warm-up, then minimum reaction, then the best available diagnostic
    pair. Actual calibration observations remain the training truth.
    """

    return (
        recommendation.stable_hold
        or recommendation.recommended_warmup
        or recommendation.minimum_reaction
        or recommendation.best_available
    )


def calculation_input_for_candidate(
    calc_input: CalculationInput,
    candidate: PowerDistanceCandidate | None,
) -> CalculationInput:
    """Return a calculation input using the selected recommendation candidate."""

    if candidate is None:
        return calc_input

    return _with_candidate_power_and_distance(
        calc_input=calc_input,
        distance=candidate.distance,
        power_percent=candidate.power_percent,
    )


def build_power_distance_recommendation(
    calc_input: CalculationInput,
    heads: dict[str, HeadConfig],
    modules: dict[str, ModuleConfig],
) -> PowerDistanceRecommendation:
    scan_input = build_recommendation_scan_input(calc_input)
    current_distance = float(scan_input.rock.distance)
    min_scan_distance = max(1, int(round(current_distance)) - 20)
    max_scan_distance = int(round(current_distance)) + 20
    candidates = scan_power_distance_grid(
        scan_input,
        heads=heads,
        modules=modules,
        min_distance=min_scan_distance,
        max_distance=max_scan_distance,
    )

    if not candidates:
        return PowerDistanceRecommendation(
            minimum_reaction=None,
            recommended_warmup=None,
            stable_hold=None,
            upper_safe=None,
            scanned_count=0,
            note="No enabled beams; enable at least one beam to calculate recommendations.",
            best_available=None,
            limiting_reason="no_enabled_beams",
            required_multiplier=None,
        )

    minimum_reaction = choose_minimum_reaction(candidates, current_distance=current_distance)
    recommended_warmup = choose_recommended_warmup(
        candidates,
        current_distance=current_distance,
        minimum_reaction=minimum_reaction,
    )
    stable = choose_stable_hold(
        candidates,
        current_distance=current_distance,
        recommended_warmup=recommended_warmup,
        minimum_reaction=minimum_reaction,
    )
    upper_safe = choose_upper_safe(candidates, current_distance=current_distance)
    best_available = max(
        candidates,
        key=lambda candidate: (candidate.effective_power, candidate.margin, -candidate.distance),
    )

    limiting_reason = "ok"
    required_multiplier = None
    if best_available.margin < 0:
        limiting_reason = "not_enough_max_power"
        required_multiplier = round(
            best_available.required_power / max(best_available.effective_power, 1.0),
            3,
        )
    elif minimum_reaction is None and stable is None:
        limiting_reason = "enough_power_but_no_safe_pair"
    elif minimum_reaction is None:
        limiting_reason = "no_minimum_reaction_pair"
    elif recommended_warmup is None:
        limiting_reason = "no_recommended_warmup_pair"
    elif stable is None:
        limiting_reason = "no_stable_hold_pair"

    return PowerDistanceRecommendation(
        minimum_reaction=minimum_reaction,
        recommended_warmup=recommended_warmup,
        stable_hold=stable,
        upper_safe=upper_safe,
        scanned_count=len(candidates),
        note=(
            "Heuristic helper: scans 20-100% power around the current scan distance "
            "using enabled beam slots and selected active modules. It splits the old "
            "warm-up hint into minimum reaction, practical warm-up, stable hold, and "
            "upper safe. Minimum reaction may only warm slowly; use recommended warm-up "
            "to build charge; stable hold may be slightly below first reaction because it is for maintaining an already-warmed rock."
        ),
        best_available=best_available,
        limiting_reason=limiting_reason,
        required_multiplier=required_multiplier,
    )
