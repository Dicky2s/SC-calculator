# Block 28 — Streamlit Arrow-safe tables

Streamlit renders `st.dataframe` through PyArrow. Detail tables often have a
generic `value` column that mixes floats, strings, booleans, dicts, and lists.
PyArrow can fail when it infers the wrong type for such a mixed column.

This block adds `sc_mining.ui.table_utils.make_arrow_safe_dataframe` and routes
UI dataframe rendering through `display_safe_dataframe`.

Numeric dataframe columns stay numeric. Mixed object columns are stringified only
for display, not in the stored JSONL/CSV dataset.
