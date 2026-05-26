from sc_mining.domain.models import ResourceComponent
from sc_mining.domain.resource_math import (
    build_resource_scu_preview_rows,
    estimate_refined_value,
    estimate_scu_from_percent,
)
from sc_mining.domain.resource_prices import get_resource_price, normalize_resource_name


def test_estimate_scu_from_percent():
    assert estimate_scu_from_percent(28.0, 66.36) == 18.581


def test_estimate_refined_value():
    assert estimate_refined_value(18.581, 481) == 8937


def test_build_resource_scu_preview_rows_includes_market_value_and_total():
    resources = [
        ResourceComponent(resource_name="titanium", resource_percent=66.36, raw_scu_estimate=None),
        ResourceComponent(resource_name="other", resource_percent=19.13, raw_scu_estimate=None),
    ]
    prices = {"titanium": 481, "other": None}

    rows = build_resource_scu_preview_rows(resources, total_scu_estimate=28.0, prices=prices)

    assert rows[0]["estimated_scu_from_percent"] == 18.581
    assert rows[0]["market_price_auec_per_scu"] == 481
    assert rows[0]["estimated_processed_value_auec"] == 8937
    assert rows[-1]["resource_name"] == "TOTAL"
    assert rows[-1]["scan_percent"] == 85.49
    assert rows[-1]["estimated_processed_value_auec"] == 8937


def test_resource_name_normalization_for_prices():
    prices = {"inert_material": 0, "titanium": 481}

    assert normalize_resource_name("Titanium") == "titanium"
    assert get_resource_price("Titanium", prices) == 481


def test_inert_materials_alias_uses_inert_material_price():
    prices = {"inert_material": 0}

    assert normalize_resource_name("inert_materials") == "inert_material"
    assert get_resource_price("inert_materials", prices) == 0
