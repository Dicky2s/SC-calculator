import pandas as pd

from sc_mining.ui.table_utils import make_arrow_safe_dataframe


def test_make_arrow_safe_dataframe_stringifies_mixed_value_column():
    df = pd.DataFrame(
        [
            {"metric": "required_power", "value": 58.3},
            {"metric": "verdict", "value": "take"},
        ]
    )

    safe = make_arrow_safe_dataframe(df)

    assert list(safe["value"]) == ["58.3", "take"]


def test_make_arrow_safe_dataframe_preserves_numeric_columns():
    df = pd.DataFrame({"mass": [1000.0, 2000.0], "margin": [1.2, -0.4]})

    safe = make_arrow_safe_dataframe(df)

    assert str(safe["mass"].dtype).startswith("float")
    assert str(safe["margin"].dtype).startswith("float")


def test_make_arrow_safe_dataframe_serializes_dict_values():
    df = pd.DataFrame(
        [
            {"metric": "payload", "value": {"verdict": "take"}},
        ]
    )

    safe = make_arrow_safe_dataframe(df)

    assert safe.iloc[0]["value"] == '{"verdict": "take"}'
