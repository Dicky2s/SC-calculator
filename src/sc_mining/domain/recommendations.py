from __future__ import annotations

from dataclasses import asdict, dataclass

from sc_mining.domain.calculator import calculate
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
    minimum_warmup: PowerDistanceCandidate | None
    stable_hold: PowerDistanceCandidate | None
    scanned_count: int
    note: str


def _candidate_to_dict(candidate: PowerDistanceCandidate | None) -> dict | None:
    return None if candidate is None else asdict(candidate)


def recommendation_to_dict(recommendation: PowerDistanceRecommendation) -> dict:
    return {
        "minimum_warmup": _candidate_to_dict(recommendation.minimum_warmup),
        "stable_hold": _candidate_to_dict(recommendation.stable_hold),
        "scanned_count": recommendation.scanned_count,
        "note": recommendation.note,
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

    for distance in range(min_distance, max_distance + 1, distance_step):
        for power in range(min_power, max_power + 1, power_step):
            candidate_input = _with_candidate_power_and_distance(
                calc_input=calc_input,
                distance=float(distance),
                power_percent=float(power),
            )
            result = calculate(candidate_input, heads=heads, modules=modules)
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


def choose_minimum_warmup(candidates: list[PowerDistanceCandidate]) -> PowerDistanceCandidate | None:
    """Find the gentlest pair that can start a successful warm-up.

    We require non-negative margin and avoid pairs the current rule-based risk
    logic marks as skip. Sorting favors lower power first, then safer/lower risk,
    then longer distance to avoid close-range overshoot.
    """

    valid = [
        candidate
        for candidate in candidates
        if candidate.margin >= 0 and candidate.risk_score < 0.75
    ]
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


def choose_stable_hold(candidates: list[PowerDistanceCandidate]) -> PowerDistanceCandidate | None:
    """Find a comfortable pair for holding the laser roughly in one place.

    Target band is intentionally conservative: slightly above required power but
    not too far above it. This should be refined by real gameplay data later.
    """

    target_margin_ratio = 0.12
    valid = [
        candidate
        for candidate in candidates
        if 0.03 <= candidate.margin_ratio <= 0.25
        and candidate.risk_score < 0.45
        and candidate.verdict == "take"
    ]
    if not valid:
        fallback = [
            candidate
            for candidate in candidates
            if candidate.margin >= 0 and candidate.risk_score < 0.60
        ]
        valid = fallback

    if not valid:
        return None

    return sorted(
        valid,
        key=lambda candidate: (
            abs(candidate.margin_ratio - target_margin_ratio),
            candidate.risk_score,
            abs(candidate.distance - 25.0),
            candidate.power_percent,
        ),
    )[0]


def build_recommendation_scan_input(calc_input: CalculationInput) -> CalculationInput:
    """Create a scan input that ignores the current UI power slider value.

    The recommendation helper is supposed to find a good power percent, so it
    must not depend on the already-selected power slider. It keeps the selected
    rock, build, enabled/selected active modules by slot, and then lets the grid
    scan replace power from 20% to 100%. When no beam is currently enabled in the
    UI, all build slots are still scanned with no active modules so the helper
    remains useful.
    """

    active_modules_by_slot = {
        beam.slot: list(beam.active_modules)
        for beam in calc_input.beams
    }

    beams = [
        BeamState(
            slot=head.slot,
            power_percent=20.0,
            active_modules=active_modules_by_slot.get(head.slot, []),
        )
        for head in calc_input.build.heads
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


def build_power_distance_recommendation(
    calc_input: CalculationInput,
    heads: dict[str, HeadConfig],
    modules: dict[str, ModuleConfig],
) -> PowerDistanceRecommendation:
    scan_input = build_recommendation_scan_input(calc_input)
    candidates = scan_power_distance_grid(scan_input, heads=heads, modules=modules)

    if not candidates:
        return PowerDistanceRecommendation(
            minimum_warmup=None,
            stable_hold=None,
            scanned_count=0,
            note="No enabled beams; enable at least one beam to calculate recommendations.",
        )

    minimum = choose_minimum_warmup(candidates)
    stable = choose_stable_hold(candidates)

    return PowerDistanceRecommendation(
        minimum_warmup=minimum,
        stable_hold=stable,
        scanned_count=len(candidates),
        note=(
            "Heuristic helper: scans 10-120m and 20-100% power using the selected "
            "build and active modules. It deliberately ignores the current UI power slider, "
            "Use as a starting hint, then calibrate with real outcomes."
        ),
    )
