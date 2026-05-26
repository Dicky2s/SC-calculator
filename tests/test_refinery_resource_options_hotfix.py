from pathlib import Path


def test_refinery_form_does_not_reference_local_resource_options():
    source = Path("src/sc_mining/ui/streamlit_app.py").read_text(encoding="utf-8")

    start = source.index("def render_refinery_form")
    end = source.find("\n\ndef ", start + 1)
    if end == -1:
        end = len(source)

    block = source[start:end]

    assert "options=resource_options" not in block
    assert "options=FALLBACK_RESOURCE_OPTIONS" in block


def test_resource_form_still_uses_profile_resource_options():
    source = Path("src/sc_mining/ui/streamlit_app.py").read_text(encoding="utf-8")

    start = source.index("def render_resource_yield_form")
    end = source.find("\n\ndef ", start + 1)
    block = source[start:end]

    assert "resource_options = resource_options_from_profiles(resource_profiles)" in block
    assert "options=resource_options" in block


def test_refinery_update_queue_does_not_reference_local_resource_options():
    source = Path("src/sc_mining/ui/streamlit_app.py").read_text(encoding="utf-8")

    start = source.index("def render_refinery_update_queue")
    end = source.find("\n\ndef ", start + 1)
    if end == -1:
        end = len(source)

    block = source[start:end]

    assert "options=resource_options" not in block
    assert "options=FALLBACK_RESOURCE_OPTIONS" in block
