from pathlib import Path


APP = Path("src/sc_mining/ui/streamlit_app.py")


def test_global_refresh_button_clears_caches_and_reruns_app() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "def refresh_streamlit_app()" in source
    assert "st.cache_data.clear()" in source
    assert "st.cache_resource.clear()" in source
    assert "st.rerun()" in source
    assert "Refresh app / reload data" in source


def test_saved_events_tab_has_local_refresh_button() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "Refresh events" in source
    assert "saved_events_refresh_events" in source
    assert "Reload saved events from the JSONL file" in source
