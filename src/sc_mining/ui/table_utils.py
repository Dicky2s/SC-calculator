import json

import pandas as pd


def make_arrow_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a Streamlit/PyArrow-safe dataframe for display.

    Numeric columns stay numeric. Mixed object columns are stringified because
    PyArrow cannot reliably infer one Arrow type from values like 58.3 + "take"
    in the same column.
    """
    safe = df.copy()

    for column in safe.columns:
        series = safe[column]

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_bool_dtype(series):
            non_null = series.dropna()
            types = {type(value) for value in non_null}

            if len(types) > 1 or any(item in types for item in (dict, list, tuple, set)):
                safe[column] = series.apply(
                    lambda value: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple, set))
                    else ("" if pd.isna(value) else str(value))
                )
            elif types and next(iter(types)) is str:
                safe[column] = series.fillna("").astype(str)

    return safe
