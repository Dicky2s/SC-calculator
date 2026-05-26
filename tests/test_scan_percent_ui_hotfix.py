from pathlib import Path

from sc_mining.domain.models import RockInput


def test_rock_input_accepts_normalized_scan_percent_instability():
    rock = RockInput(mass=12600, resistance=0.0, instability=0.30, distance=15)
    assert rock.instability == 0.30


def test_streamlit_ui_uses_scan_percent_fields_and_default_15m():
    source = Path("src/sc_mining/ui/streamlit_app.py").read_text(encoding="utf-8")

    assert '"Resistance, %"' in source
    assert '"Instability, %"' in source
    assert "max_value=1000.0" in source
    assert "resistance = scan_percent_to_fraction(resistance_percent)" in source
    assert "instability = scan_percent_to_fraction(instability_percent)" in source
    assert 'key=f"{key_prefix}_distance"' in source
    assert "value=15.0" in source


def test_rock_input_accepts_high_instability_scan_fraction():
    rock = RockInput(mass=31000, resistance=0.43, instability=2.8474, distance=15)
    assert rock.instability == 2.8474
