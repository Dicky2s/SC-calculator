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


def test_sidebar_build_selection_uses_versioned_ship_scoped_widget_keys() -> None:
    source = APP.read_text(encoding="utf-8")

    assert 'sidebar_state_version = "v5_visible_prospector"' in source
    assert 'key=f"selected_ship_type__{sidebar_state_version}"' in source
    assert 'selected_build_state_key = f"selected_build_profile_path__{sidebar_state_version}"' in source
    assert 'f"selected_build_profile_path__{sidebar_state_version}__{selected_ship}"' in source
    assert "selected_build_widget_key = (" in source
    assert "selected_build_profile_widget__{sidebar_state_version}__{selected_ship}__{module_filter_token}" in source
    assert "filtered_build_options," in source
    assert "selected_build_path = st.sidebar.selectbox(" in source
    assert "st.session_state[selected_build_state_key] = selected_build_path" in source
    assert "st.session_state[selected_ship_build_state_key] = selected_build_path" in source
    assert "remembered_build not in filtered_build_options" in source
    assert "selected_build.ship_type != selected_ship" in source
    assert 'key=selected_build_widget_key' in source


def test_sidebar_build_selection_purges_legacy_stale_widget_keys() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "legacy_sidebar_state_keys = [" in source
    assert 'key == "selected_ship_type"' in source
    assert 'key == "selected_build_profile_path"' in source
    assert 'key.startswith("selected_build_profile_path__")' in source
    assert 'key.startswith("selected_build_profile_index__")' in source
    assert 'key.startswith("selected_build_module_filter__")' in source
    assert "del st.session_state[key]" in source


def test_sidebar_build_selection_has_module_filter() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "def build_module_ids(build) -> set[str]:" in source
    assert "def build_matches_module_filter(build, selected_module_ids: list[str]) -> bool:" in source
    assert 'st.sidebar.multiselect(' in source
    assert '"Filter by modules"' in source
    assert "build_matches_module_filter(build_by_path[path], selected_module_ids)" in source


def test_sidebar_loadout_is_rendered_as_plain_text() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "for line in build_loadout_text_lines(build, heads, modules):" in source
    assert 'st.markdown(f"- {line}")' in source
    assert "pd.DataFrame(build_loadout_rows(build, modules))" not in source
    assert "Loadout: **{build_profile_label(build, heads, modules)}**" in source
    assert 'with st.sidebar.expander("Loadout details", expanded=False):' in source


def test_sidebar_uses_temporary_visible_build_whitelist() -> None:
    source = APP.read_text(encoding="utf-8")

    assert "VISIBLE_BUILD_PROFILE_FILENAMES" in source
    assert "prospector_helix_2x_rieger.yaml" in source
    assert "prospector_helix_rieger_torrent_iii.yaml" in source
    assert "build_files = list_visible_build_files()" in source
    assert "Visible build profiles" in source
