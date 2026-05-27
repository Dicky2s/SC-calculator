from pathlib import Path


APP = Path("src/sc_mining/ui/streamlit_app.py")


def test_build_label_helpers_render_readable_text() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "def module_display_name(" in source
    assert "def head_display_name(" in source
    assert "def compact_module_list(" in source
    assert "def build_profile_label(build, heads: dict, modules: dict)" in source
    assert "f\"{counts[name]}× {name}\"" in source
    assert "return f\"{ship_name} — {slot_summary}\"" in source


def test_sidebar_build_selection_uses_persistent_session_state_key() -> None:
    source = APP.read_text(encoding="utf-8")

    assert 'selected_build_state_key = "selected_build_profile_path"' in source
    assert 'selected_ship_build_state_key = f"selected_build_profile_path__{selected_ship}"' in source
    assert "st.session_state[selected_build_state_key] = remembered_build" in source
    assert "st.session_state[selected_ship_build_state_key] = selected_build_path" in source
    assert "remembered_build not in filtered_build_options" in source
    assert 'key=selected_build_state_key' in source


def test_sidebar_loadout_is_rendered_as_plain_text() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "for line in build_loadout_text_lines(build, heads, modules):" in source
    assert 'st.markdown(f"- {line}")' in source
    assert "pd.DataFrame(build_loadout_rows(build, modules))" not in source
    assert "Loadout: **{build_profile_label(build, heads, modules)}**" in source
