from __future__ import annotations

from sc_mining.domain.models import ResourceComponent
from sc_mining.domain.resource_prices import get_resource_price, normalize_resource_name


def aggregate_resources_by_name(resources: list[ResourceComponent]) -> list[ResourceComponent]:
    """Aggregate duplicate resource rows for previews and summary features.

    The raw UI can keep several scan lines for the same material, but totals,
    market value and dominant-resource selection should treat them as one
    resource. Percent and SCU are summed; qualitative hints are preserved only
    when all duplicate rows agree, otherwise they become ``unknown``.
    """

    grouped: dict[str, dict] = {}

    for item in resources:
        key = normalize_resource_name(item.resource_name or "unknown")
        if key not in grouped:
            grouped[key] = {
                "resource_name": item.resource_name or key,
                "resource_percent": 0.0,
                "has_percent": False,
                "raw_scu_estimate": 0.0,
                "has_raw_scu": False,
                "observed_window_sizes": set(),
                "observed_charge_behaviors": set(),
                "comments": [],
            }

        bucket = grouped[key]
        if item.resource_percent is not None:
            bucket["resource_percent"] += float(item.resource_percent)
            bucket["has_percent"] = True
        if item.raw_scu_estimate is not None:
            bucket["raw_scu_estimate"] += float(item.raw_scu_estimate)
            bucket["has_raw_scu"] = True

        if item.observed_window_size and item.observed_window_size != "unknown":
            bucket["observed_window_sizes"].add(item.observed_window_size)
        if item.observed_charge_behavior and item.observed_charge_behavior != "unknown":
            bucket["observed_charge_behaviors"].add(item.observed_charge_behavior)
        if item.comment.strip():
            bucket["comments"].append(item.comment.strip())

    aggregated: list[ResourceComponent] = []
    for bucket in grouped.values():
        window_values = bucket["observed_window_sizes"]
        charge_values = bucket["observed_charge_behaviors"]
        aggregated.append(
            ResourceComponent(
                resource_name=bucket["resource_name"],
                resource_percent=(
                    round(bucket["resource_percent"], 3) if bucket["has_percent"] else None
                ),
                raw_scu_estimate=(
                    round(bucket["raw_scu_estimate"], 3) if bucket["has_raw_scu"] else None
                ),
                observed_window_size=next(iter(window_values)) if len(window_values) == 1 else "unknown",
                observed_charge_behavior=next(iter(charge_values)) if len(charge_values) == 1 else "unknown",
                comment=" | ".join(bucket["comments"]),
            )
        )

    return aggregated


def estimate_scu_from_percent(
    total_scu_estimate: float | None,
    resource_percent: float | None,
) -> float | None:
    if total_scu_estimate is None or total_scu_estimate <= 0:
        return None
    if resource_percent is None:
        return None

    return round(float(total_scu_estimate) * float(resource_percent) / 100.0, 3)


def estimate_refined_value(
    scu_estimate: float | None,
    price_auec_per_scu: float | None,
) -> float | None:
    if scu_estimate is None or price_auec_per_scu is None:
        return None

    return round(float(scu_estimate) * float(price_auec_per_scu), 0)


def build_resource_scu_preview_rows(
    resources: list[ResourceComponent],
    total_scu_estimate: float | None,
    prices: dict[str, float | None] | None = None,
) -> list[dict]:
    prices = prices or {}
    resources = aggregate_resources_by_name(resources)

    rows: list[dict] = []
    total_percent = 0.0
    total_estimated_scu = 0.0
    total_estimated_value = 0.0

    for item in resources:
        percent = item.resource_percent
        estimated_scu = estimate_scu_from_percent(total_scu_estimate, percent)
        saved_raw_scu = item.raw_scu_estimate

        # Prefer explicit saved SCU if user manually entered it; otherwise use derived SCU.
        scu_for_value = saved_raw_scu if saved_raw_scu not in (None, 0, 0.0) else estimated_scu
        price = get_resource_price(item.resource_name, prices)
        estimated_value = estimate_refined_value(scu_for_value, price)

        if percent is not None:
            total_percent += float(percent)
        if estimated_scu is not None:
            total_estimated_scu += float(estimated_scu)
        if estimated_value is not None:
            total_estimated_value += float(estimated_value)

        rows.append(
            {
                "resource_name": item.resource_name,
                "scan_percent": percent,
                "estimated_scu_from_percent": estimated_scu,
                "saved_raw_scu": saved_raw_scu,
                "market_price_auec_per_scu": price,
                "estimated_processed_value_auec": estimated_value,
                "comment": item.comment,
            }
        )

    if rows:
        rows.append(
            {
                "resource_name": "TOTAL",
                "scan_percent": round(total_percent, 3),
                "estimated_scu_from_percent": round(total_estimated_scu, 3),
                "saved_raw_scu": "",
                "market_price_auec_per_scu": "",
                "estimated_processed_value_auec": round(total_estimated_value, 0),
                "comment": "",
            }
        )

    return rows
