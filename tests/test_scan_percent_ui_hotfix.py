from pathlib import Path

import pytest
from pydantic import ValidationError

from sc_mining.domain.models import RockInput


def test_rock_input_rejects_raw_scan_percent_instability():
    with pytest.raises(ValidationError):
        RockInput(mass=12600, resistance=0.0, instability=30.0, distance=15)


def test_rock_input_accepts_normalized_scan_percent_instability():
    rock = RockInput(mass=12600, resistance=0.0, instability=0.30, distance=15)
    assert rock.instability == 0.30


def test_streamlit_ui_uses_scan_percent_fields_and_default_15m():
    source = Path("src/sc_mining/ui/streamlit_app.py").read_text(encoding="utf-8")

    assert '"Resistance, %"' in source
    assert '"Instability, %"' in source
    assert "resistance = scan_percent_to_fraction(resistance_percent)" in source
    assert "instability = scan_percent_to_fraction(instability_percent)" in source
    assert 'key=f"{key_prefix}_distance"' in source
    assert "value=15.0" in source
