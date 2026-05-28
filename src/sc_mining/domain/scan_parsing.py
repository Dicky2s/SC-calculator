from __future__ import annotations

import math
import re
from typing import Any


_THOUSANDS_PATTERN = re.compile(r"^\d{1,3}([.,\s]\d{3})+$")
_NON_NUMERIC_PREFIX = re.compile(r"^(mass|масса)\s*[:=]?\s*", re.IGNORECASE)


def _normalize_raw_text(value: str) -> str:
    cleaned = str(value).strip()
    cleaned = _NON_NUMERIC_PREFIX.sub("", cleaned)
    cleaned = cleaned.replace("\u00a0", " ").replace("'", "").replace("_", "")
    return cleaned.strip()


def _auto_correct_numeric_mass(value: float) -> float:
    """Correct common scan mass input mistakes.

    In Star Citizen scan UI, rock mass is usually a thousands-style integer value.
    Users often type values copied from the UI as ``4.666`` or ``4,666`` meaning
    4666, while percentages like ``22,71`` must stay decimal percentages and are
    parsed elsewhere. This helper is intentionally mass-only.
    """

    if not math.isfinite(value) or value <= 0:
        raise ValueError("Mass must be a positive finite number.")

    # Numeric callers may already have converted '4.666' into float(4.666).
    # Treat a small value with exactly three meaningful decimal places as a
    # thousands-separated mass. This keeps '710.00' as 710.0 and fixes 4.666.
    if 1.0 <= value < 100.0:
        scaled = value * 1000.0
        if abs(round(scaled) - scaled) < 1e-6:
            return float(round(scaled))

    return float(value)


def parse_scan_mass_value(value: Any) -> float:
    """Parse the in-game Mass field without confusing thousands and decimals.

    Mass examples:
    - ``4.666`` -> 4666
    - ``4,666`` -> 4666
    - ``23.295`` -> 23295
    - ``23,295`` -> 23295
    - ``710,00`` -> 710.0

    Percentage values are intentionally not handled here; use the existing
    percent parser for resistance/instability.
    """

    if isinstance(value, (int, float)):
        return _auto_correct_numeric_mass(float(value))

    raw = _normalize_raw_text(str(value))
    compact = raw.replace(" ", "")

    if not compact:
        raise ValueError("Mass is empty.")

    if _THOUSANDS_PATTERN.fullmatch(raw):
        integer_text = re.sub(r"[.,\s]", "", raw)
        return _auto_correct_numeric_mass(float(integer_text))

    # Decimal number, using comma or dot as decimal separator.
    decimal_text = compact.replace(",", ".")
    try:
        parsed = float(decimal_text)
    except ValueError as exc:
        raise ValueError(f"Could not parse mass value: {value!r}") from exc

    return _auto_correct_numeric_mass(parsed)


def describe_scan_mass_parse(raw_value: Any, parsed_mass: float) -> str:
    """Return a short UI note when mass was normalized from a compact scan format."""

    raw = _normalize_raw_text(str(raw_value))
    compact = raw.replace(" ", "")
    if not compact:
        return ""

    try:
        naive = float(compact.replace(",", "."))
    except ValueError:
        naive = None

    if naive is not None and naive > 0 and abs(naive - parsed_mass) > 1e-9:
        return f"Mass parsed as {parsed_mass:g} from input {raw!r}."

    if parsed_mass < 100.0:
        return "Mass is below 100. Check the scan value; this usually means a thousands separator was entered incorrectly."

    return ""


def is_valid_training_mass(mass: Any, minimum_mass: float = 100.0) -> bool:
    """Return whether a mass value is safe enough to use for real ML training."""

    try:
        parsed = float(mass)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed >= minimum_mass
