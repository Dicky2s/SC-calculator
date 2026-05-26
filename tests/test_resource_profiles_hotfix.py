from pathlib import Path

from sc_mining.domain.config_loader import load_resources


def test_resources_config_constant_exists_in_streamlit_app():
    source = Path("src/sc_mining/ui/streamlit_app.py").read_text(encoding="utf-8")

    assert 'RESOURCES_CONFIG = Path("configs") / "resources.yaml"' in source
    assert "resource_profiles = load_resources(RESOURCES_CONFIG)" in source
    assert "resource_profiles=resource_profiles" in source


def test_resources_yaml_loads():
    resources = load_resources("configs/resources.yaml")

    assert "beryl" in resources
    assert "copper" in resources
    assert resources["beryl"].label
