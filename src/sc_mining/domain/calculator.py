from __future__ import annotations

from sc_mining.domain.models import (
    BeamState,
    BuildProfile,
    CalculationInput,
    CalculationResult,
    HeadConfig,
    ModuleConfig,
    RockInput,
)


MIN_BEAM_POWER_PERCENT = 20.0
RAW_POWER_SCALE = 3150.0
DEFAULT_OPTIMAL_RANGE_METERS = 15.0
DEFAULT_MAX_RANGE_METERS = 45.0
MIN_DISTANCE_EFFICIENCY_AT_MAX_RANGE = 0.35
STABLE_TAKE_MARGIN_RATIO = 0.10
ALMOST_MARGIN_RATIO = 0.90
RAW_REQUIRED_POWER_PER_MASS = 0.20
RESISTANCE_REQUIRED_POWER_WEIGHT = 1.0



def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def head_max_power(head: HeadConfig) -> float:
    """Return raw max mining power for a head.

    Older configs stored `base_power` as a normalized multiplier where Helix S1
    was 1.00. Newer configs can provide `max_power` directly. Keeping both lets
    old profiles still load while the calculator works in raw power units.
    """
    if head.max_power is not None:
        return float(head.max_power)

    return float(head.base_power) * RAW_POWER_SCALE


def head_optimal_range(head: HeadConfig) -> float:
    return float(head.optimal_range or DEFAULT_OPTIMAL_RANGE_METERS)


def head_max_range(head: HeadConfig) -> float:
    return float(head.max_range or DEFAULT_MAX_RANGE_METERS)


def validate_distance_for_head(distance: float, head: HeadConfig) -> None:
    if distance <= 0:
        raise ValueError("Distance must be greater than 0")


def distance_efficiency_for_head(distance: float, head: HeadConfig) -> float:
    """Approximate laser delivery loss by range.

    Important calibration rules:
    - no distance bonus above 1.0;
    - at/below optimal range the factor is 1.0;
    - beyond optimal range the factor falls linearly toward a conservative
      residual efficiency at max range;
    - beyond max range the event is invalid instead of being calculated.
    """
    safe_distance = float(distance)
    validate_distance_for_head(safe_distance, head)

    optimal = head_optimal_range(head)
    max_range = head_max_range(head)

    if safe_distance <= optimal:
        return 1.0

    if max_range <= optimal:
        return 1.0

    capped_distance = min(safe_distance, max_range)
    t = (capped_distance - optimal) / (max_range - optimal)
    factor = 1.0 - t * (1.0 - MIN_DISTANCE_EFFICIENCY_AT_MAX_RANGE)

    return clamp(factor, MIN_DISTANCE_EFFICIENCY_AT_MAX_RANGE, 1.0)


def distance_efficiency(distance: float) -> float:
    """Backward-compatible helper using the default 15m/45m S1 range profile."""
    dummy = HeadConfig(
        name="Default mining head",
        size=1,
        base_power=1.0,
        max_power=RAW_POWER_SCALE,
        optimal_range=DEFAULT_OPTIMAL_RANGE_METERS,
        max_range=DEFAULT_MAX_RANGE_METERS,
    )
    return distance_efficiency_for_head(distance, dummy)


def calculate_required_power(rock: RockInput) -> float:
    """Approximate raw power needed for first rock reaction.

    Earlier builds used ``mass * 0.2 / (1 - resistance)``. Real calibration
    rows showed that this over-pushed resistant rocks: the helper could suggest
    values that reacted, but then immediately overheated/climbed too hard.

    The current field calibration uses a gentler resistance response:
        required = mass * 0.20 * (1 + resistance)

    `resistance` is stored as a fraction in this app: 47% -> 0.47.
    Instability affects risk/control, not the basic first-reaction threshold.
    Distance affects delivered beam power, not rock difficulty.
    """
    if rock.resistance < 0:
        raise ValueError("Rock resistance must not be negative")

    resistance_factor = 1.0 + float(rock.resistance) * RESISTANCE_REQUIRED_POWER_WEIGHT
    return float(rock.mass) * RAW_REQUIRED_POWER_PER_MASS * resistance_factor


def find_head_build(build: BuildProfile, slot: str):
    for head in build.heads:
        if head.slot == slot:
            return head

    raise ValueError(f"Unknown beam slot in build: {slot}")


def calculate_build_power_multiplier(
    module_ids: list[str],
    modules: dict[str, ModuleConfig],
) -> float:
    power_multiplier = 1.0

    for module_id in module_ids:
        if module_id not in modules:
            raise ValueError(f"Unknown module_id in build: {module_id}")

        module = modules[module_id]
        if module.type == "passive":
            power_multiplier *= module.power_modifier

    return power_multiplier


def calculate_active_power_multiplier(
    module_ids: list[str],
    modules: dict[str, ModuleConfig],
    notes: list[str],
) -> float:
    power_multiplier = 1.0

    for module_id in module_ids:
        if module_id not in modules:
            raise ValueError(f"Unknown active module_id: {module_id}")

        module = modules[module_id]

        if module.type != "active":
            notes.append(f"Module {module_id} is not active type, ignored as active")
            continue

        power_multiplier *= module.power_modifier
        notes.append(f"Active module applied: {module_id}")

    return power_multiplier


def calculate_beam_power(
    beam: BeamState,
    build: BuildProfile,
    heads: dict[str, HeadConfig],
    modules: dict[str, ModuleConfig],
    distance: float,
) -> tuple[float, list[str]]:
    notes: list[str] = []

    head_build = find_head_build(build, beam.slot)

    if head_build.head_id not in heads:
        raise ValueError(f"Unknown head_id in build: {head_build.head_id}")

    head = heads[head_build.head_id]
    validate_distance_for_head(distance, head)

    passive_multiplier = calculate_build_power_multiplier(head_build.modules, modules)
    active_multiplier = calculate_active_power_multiplier(beam.active_modules, modules, notes)

    if beam.power_percent < MIN_BEAM_POWER_PERCENT:
        raise ValueError(
            f"Beam {beam.slot} power_percent must be at least "
            f"{MIN_BEAM_POWER_PERCENT:.0f}. Mining laser UI starts at 20%."
        )

    full_power = head_max_power(head) * passive_multiplier * active_multiplier
    power_before_distance = full_power * (beam.power_percent / 100.0)
    efficiency = distance_efficiency_for_head(distance, head)
    power_after_distance = power_before_distance * efficiency

    notes.append(
        f"Beam {beam.slot}: head={head_build.head_id}, "
        f"full_power={full_power:.2f}, power_percent={beam.power_percent}, "
        f"effective_before_distance={power_before_distance:.2f}"
    )
    notes.append(
        f"Distance efficiency for {beam.slot}: distance={distance}, "
        f"optimal={head_optimal_range(head):.1f}, max={head_max_range(head):.1f}, "
        f"factor={efficiency:.3f}, effective_after_distance={power_after_distance:.2f}"
    )

    return power_after_distance, notes


def calculate_effective_power(
    calc_input: CalculationInput,
    heads: dict[str, HeadConfig],
    modules: dict[str, ModuleConfig],
) -> tuple[float, list[str]]:
    total = 0.0
    notes: list[str] = []

    if not calc_input.beams:
        return 0.0, ["No active beams provided"]

    for beam in calc_input.beams:
        beam_power, beam_notes = calculate_beam_power(
            beam=beam,
            build=calc_input.build,
            heads=heads,
            modules=modules,
            distance=calc_input.rock.distance,
        )
        total += beam_power
        notes.extend(beam_notes)

    notes.append(f"Total effective power: {total:.2f}")

    return total, notes


def calculate_risk_score(
    required_power: float,
    effective_power: float,
    rock: RockInput,
) -> float:
    """Risk score in range 0..1.

    High instability, low/negative margin, and large overpower all increase risk.
    The value is diagnostic; verdict boundaries are handled separately.
    """
    if effective_power <= 0:
        return 1.0

    margin_ratio = (effective_power - required_power) / max(required_power, 1.0)

    if margin_ratio < 0:
        margin_risk = clamp(abs(margin_ratio), 0.0, 1.0)
    elif margin_ratio < STABLE_TAKE_MARGIN_RATIO:
        margin_risk = 0.45 - 0.25 * (margin_ratio / STABLE_TAKE_MARGIN_RATIO)
    else:
        margin_risk = 0.10

    instability_risk = clamp(float(rock.instability), 0.0, 1.0) * 0.45

    overpower_ratio = effective_power / max(required_power, 1.0)
    overpower_risk = clamp((overpower_ratio - 1.35) / 1.50, 0.0, 1.0) * 0.35

    return clamp(margin_risk + instability_risk + overpower_risk, 0.0, 1.0)


def choose_verdict(required_power: float, effective_power: float, risk_score: float) -> str:
    if effective_power < required_power * ALMOST_MARGIN_RATIO:
        return "need_more_power"

    if effective_power < required_power:
        return "almost"

    if effective_power < required_power * (1.0 + STABLE_TAKE_MARGIN_RATIO):
        return "edge_take"

    if risk_score >= 0.75:
        return "skip"

    if risk_score >= 0.45:
        return "risky"

    return "take"


def calculate(
    calc_input: CalculationInput,
    heads: dict[str, HeadConfig],
    modules: dict[str, ModuleConfig],
) -> CalculationResult:
    required = calculate_required_power(calc_input.rock)
    effective, notes = calculate_effective_power(calc_input, heads=heads, modules=modules)

    margin = effective - required
    risk = calculate_risk_score(required, effective, calc_input.rock)
    verdict = choose_verdict(required, effective, risk)

    return CalculationResult(
        required_power=round(required, 3),
        effective_power=round(effective, 3),
        margin=round(margin, 3),
        risk_score=round(risk, 3),
        verdict=verdict,
        notes=notes,
    )
