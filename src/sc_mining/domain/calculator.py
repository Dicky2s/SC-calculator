from sc_mining.domain.models import (
    BeamState,
    BuildProfile,
    CalculationInput,
    CalculationResult,
    HeadConfig,
    ModuleConfig,
    RockInput,
)


def distance_modifier(distance: float) -> float:
    """
    Baseline approximation:
    - no penalty up to 100m;
    - after 100m difficulty grows slowly.
    """
    if distance <= 100:
        return 1.0

    return 1.0 + ((distance - 100) / 100) * 0.15


def calculate_required_power(rock: RockInput) -> float:
    """
    Baseline formula.

    This is not the real Star Citizen formula.
    It is a first approximation for future calibration.
    """
    mass_factor = rock.mass / 1000.0
    resistance_factor = 1.0 + rock.resistance * 1.8
    instability_factor = 1.0 + rock.instability * 0.8
    dist_factor = distance_modifier(rock.distance)

    return mass_factor * resistance_factor * instability_factor * dist_factor


def find_head_build(build: BuildProfile, slot: str):
    for head in build.heads:
        if head.slot == slot:
            return head

    raise ValueError(f"Unknown beam slot in build: {slot}")


def calculate_beam_power(
    beam: BeamState,
    build: BuildProfile,
    heads: dict[str, HeadConfig],
    modules: dict[str, ModuleConfig],
) -> tuple[float, list[str]]:
    notes: list[str] = []

    head_build = find_head_build(build, beam.slot)

    if head_build.head_id not in heads:
        raise ValueError(f"Unknown head_id in build: {head_build.head_id}")

    head = heads[head_build.head_id]

    power_multiplier = 1.0

    for module_id in head_build.modules:
        if module_id not in modules:
            raise ValueError(f"Unknown module_id in build: {module_id}")

        module = modules[module_id]

        if module.type == "passive":
            power_multiplier *= module.power_modifier

    for module_id in beam.active_modules:
        if module_id not in modules:
            raise ValueError(f"Unknown active module_id: {module_id}")

        module = modules[module_id]

        if module.type != "active":
            notes.append(f"Module {module_id} is not active type, ignored as active")
            continue

        power_multiplier *= module.power_modifier
        notes.append(f"Active module applied: {module_id}")

    base = head.base_power * 100.0
    power = base * (beam.power_percent / 100.0) * power_multiplier

    notes.append(
        f"Beam {beam.slot}: head={head_build.head_id}, "
        f"power_percent={beam.power_percent}, effective={power:.2f}"
    )

    return power, notes


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
        )
        total += beam_power
        notes.extend(beam_notes)

    return total, notes


def calculate_risk_score(
    required_power: float,
    effective_power: float,
    rock: RockInput,
) -> float:
    """
    Risk score in range 0..1.

    Higher instability and lower margin increase risk.
    """
    if effective_power <= 0:
        return 1.0

    margin_ratio = (effective_power - required_power) / max(required_power, 1.0)

    margin_risk = 1.0 - max(min((margin_ratio + 0.3) / 0.8, 1.0), 0.0)
    instability_risk = min(rock.instability, 1.0) * 0.5

    return max(min(margin_risk + instability_risk, 1.0), 0.0)


def choose_verdict(margin: float, risk_score: float) -> str:
    if margin < 0:
        return "need_more_power"

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
    effective, notes = calculate_effective_power(calc_input, heads, modules)

    margin = effective - required
    risk = calculate_risk_score(required, effective, calc_input.rock)
    verdict = choose_verdict(margin, risk)

    return CalculationResult(
        required_power=round(required, 3),
        effective_power=round(effective, 3),
        margin=round(margin, 3),
        risk_score=round(risk, 3),
        verdict=verdict,
        notes=notes,
    )