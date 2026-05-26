from pathlib import Path
from typing import Any

import yaml


DEFAULT_RESOURCE_PRICE_PATH = Path("configs") / "resource_prices.yaml"


def normalize_resource_name(name: str | None) -> str:
    return str(name or "unknown").strip().lower().replace(" ", "_")


def load_resource_price_config(path: str | Path = DEFAULT_RESOURCE_PRICE_PATH) -> dict[str, Any]:
    price_path = Path(path)

    if not price_path.exists():
        return {
            "version": "missing",
            "currency": "aUEC",
            "unit": "per_scu_refined",
            "prices": {},
        }

    with price_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    prices = payload.get("prices") or {}
    normalized_prices: dict[str, float | None] = {}

    for resource_name, value in prices.items():
        normalized = normalize_resource_name(resource_name)
        normalized_prices[normalized] = float(value) if value is not None else None

    return {
        "version": payload.get("version", "unknown"),
        "currency": payload.get("currency", "aUEC"),
        "unit": payload.get("unit", "per_scu_refined"),
        "prices": normalized_prices,
    }


def load_resource_prices(path: str | Path = DEFAULT_RESOURCE_PRICE_PATH) -> dict[str, float | None]:
    return load_resource_price_config(path)["prices"]


def get_resource_price(
    resource_name: str | None,
    prices: dict[str, float | None],
) -> float | None:
    return prices.get(normalize_resource_name(resource_name))
