import pytest
from pydantic import ValidationError

from sc_mining.domain.models import RockInput
from sc_mining.ui.table_utils import make_arrow_safe_dataframe


def test_rock_input_rejects_percent_values_without_normalization():
    with pytest.raises(ValidationError):
        RockInput(mass=10000, resistance=0.0, instability=23.0, distance=15)


def test_rock_input_accepts_normalized_scan_values():
    rock = RockInput(mass=10000, resistance=0.0, instability=0.23, distance=15)

    assert rock.instability == 0.23
    assert rock.resistance == 0.0
