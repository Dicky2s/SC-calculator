from __future__ import annotations

from sc_mining.domain.models import ResourceComponent
from sc_mining.domain.resource_prices import get_resource_price


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
