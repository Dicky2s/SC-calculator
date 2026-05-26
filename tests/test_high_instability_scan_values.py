from pathlib import Path

from sc_mining.domain.models import RockInput


def test_high_instability_scan_value_is_valid_after_normalization():
    rock = RockInput(mass=31000, resistance=0.43, instability=2.8474, distance=15)

    assert rock.resistance == 0.43
    assert rock.instability == 2.8474


def test_streamlit_ui_allows_high_instability_scan_percent():
    source = Path("src/sc_mining/ui/streamlit_app.py").read_text(encoding="utf-8")

    assert '"Instability, %"' in source
    assert "max_value=1000.0" in source
    assert "scan 284.74% becomes 2.8474" in source
    assert "return max(0.0, float(value) / 100.0)" in source
