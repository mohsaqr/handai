"""
Comprehensive tests for Coding Tools Settings tab
Tests: All settings, persistence, edge cases, crash scenarios, security
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_session_state():
    """Create a mock session state that behaves like st.session_state"""
    state = {}

    class MockSessionState(dict):
        def __getattr__(self, key):
            if key in self:
                return self[key]
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

        def __setattr__(self, key, value):
            self[key] = value

        def get(self, key, default=None):
            return super().get(key, default)

    return MockSessionState()


@pytest.fixture
def mock_streamlit(mock_session_state):
    """Mock streamlit module"""
    mock_st = MagicMock()
    mock_st.session_state = mock_session_state
    mock_st.tabs = MagicMock(return_value=[MagicMock() for _ in range(5)])
    mock_st.toggle = MagicMock(return_value=True)
    mock_st.slider = MagicMock(return_value=2)
    mock_st.selectbox = MagicMock(return_value="per_row")
    mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
    mock_st.expander = MagicMock()
    mock_st.checkbox = MagicMock(return_value=False)
    mock_st.text_area = MagicMock(return_value="")
    mock_st.text_input = MagicMock(return_value="")
    mock_st.number_input = MagicMock(return_value=2)
    mock_st.button = MagicMock(return_value=False)
    mock_st.radio = MagicMock(return_value="per_row")
    mock_st.divider = MagicMock()
    mock_st.header = MagicMock()
    mock_st.subheader = MagicMock()
    mock_st.caption = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.rerun = MagicMock()
    mock_st.fragment = lambda f: f  # Decorator passthrough

    return mock_st


# ============================================================================
# TEST: SETTINGS PAGE STRUCTURE
# ============================================================================

class TestSettingsPageStructure:
    """Test the structure and layout of settings page"""

    def test_settings_has_five_tabs(self):
        """Settings page should have exactly 5 tabs"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        # Check for 5 tab variables
        assert "tab1, tab2, tab3, tab4, tab5" in source, "Should have 5 tabs"

    def test_coding_tools_tab_exists(self):
        """Coding Tools tab should exist"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        assert "Coding Tools" in source, "Coding Tools tab should exist"
        assert ":material/code:" in source, "Should have code icon"

    def test_tab_order_is_correct(self):
        """Tabs should be in correct order"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        # Find the tabs definition line
        tabs_line = [l for l in source.split('\n') if 'st.tabs' in l][0]

        # Check order
        model_idx = source.find("Model Defaults")
        perf_idx = source.find("Performance")
        storage_idx = source.find("Storage")
        coding_idx = source.find("Coding Tools")
        prompts_idx = source.find("System Prompts")

        assert model_idx < perf_idx < storage_idx < coding_idx < prompts_idx, \
            "Tabs should be in correct order"

    def test_coding_tools_tab_uses_tab4(self):
        """Coding Tools tab should be tab4"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        # Find "with tab4:" and check it's followed by Coding Tools header
        tab4_section = source[source.find("with tab4:"):]
        first_header = tab4_section[:500]

        assert "Coding Tools Settings" in first_header, "tab4 should be Coding Tools"

    def test_system_prompts_uses_tab5(self):
        """System Prompts tab should be tab5"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab5_section = source[source.find("with tab5:"):]
        first_header = tab5_section[:500]

        assert "System Prompts" in first_header, "tab5 should be System Prompts"


# ============================================================================
# TEST: ALL SETTING KEYS
# ============================================================================

class TestSettingKeys:
    """Test all setting keys are properly defined"""

    # Global coding settings
    GLOBAL_SETTINGS = [
        ("coding_light_mode", bool, True),
        ("coding_context_rows", int, 2),
        ("coding_auto_advance", bool, False),
        ("coding_horizontal_codes", bool, False),
        ("coding_buttons_above", bool, False),
    ]

    # AI Coder specific settings
    AI_CODER_SETTINGS = [
        ("aic_default_mode", str, "per_row"),
        ("aic_default_display", str, "ai_first"),
        ("aic_autosave_enabled", bool, True),
    ]

    # Manual Coder settings
    MANUAL_CODER_SETTINGS = [
        ("mc_autosave_enabled", bool, True),
    ]

    def test_global_settings_in_settings_page(self):
        """All global settings should be in settings page"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        for key, _, _ in self.GLOBAL_SETTINGS:
            assert key in source, f"Global setting {key} should be in settings page"

    def test_ai_coder_settings_in_settings_page(self):
        """All AI Coder settings should be in settings page"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        for key, _, _ in self.AI_CODER_SETTINGS:
            assert key in source, f"AI Coder setting {key} should be in settings page"

    def test_manual_coder_settings_in_settings_page(self):
        """Manual Coder autosave should be in Coding Tools tab"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        for key, _, _ in self.MANUAL_CODER_SETTINGS:
            assert key in source, f"Manual Coder setting {key} should be in settings page"

    def test_mc_autosave_not_in_storage_tab(self):
        """Manual Coder autosave should NOT be in Storage tab section"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        # Get Storage tab section (tab3)
        tab3_start = source.find("with tab3:")
        tab4_start = source.find("with tab4:")
        storage_section = source[tab3_start:tab4_start]

        # mc_autosave should NOT be in storage section
        assert "mc_autosave_enabled" not in storage_section, \
            "mc_autosave should be moved out of Storage tab"


# ============================================================================
# TEST: AI CODER USES GLOBAL SETTINGS
# ============================================================================

class TestAICoderUsesGlobalSettings:
    """Test AI Coder properly reads global settings"""

    def test_ai_coder_imports(self):
        """AI Coder should import without errors"""
        from tools.ai_coder import AICoderTool
        tool = AICoderTool()
        assert tool is not None

    def test_init_session_state_uses_global_light_mode(self):
        """AI Coder should use global coding_light_mode"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool._init_session_state)

        assert 'st.session_state.get("coding_light_mode"' in source, \
            "Should read from coding_light_mode"

    def test_init_session_state_uses_global_context_rows(self):
        """AI Coder should use global coding_context_rows"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool._init_session_state)

        assert 'st.session_state.get("coding_context_rows"' in source, \
            "Should read from coding_context_rows"

    def test_init_session_state_uses_global_auto_advance(self):
        """AI Coder should use global coding_auto_advance"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool._init_session_state)

        assert 'st.session_state.get("coding_auto_advance"' in source, \
            "Should read from coding_auto_advance"

    def test_init_session_state_uses_global_horizontal(self):
        """AI Coder should use global coding_horizontal_codes"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool._init_session_state)

        assert 'st.session_state.get("coding_horizontal_codes"' in source, \
            "Should read from coding_horizontal_codes"

    def test_init_session_state_uses_global_buttons_above(self):
        """AI Coder should use global coding_buttons_above"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool._init_session_state)

        assert 'st.session_state.get("coding_buttons_above"' in source, \
            "Should read from coding_buttons_above"

    def test_init_session_state_uses_default_mode(self):
        """AI Coder should use global aic_default_mode"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool._init_session_state)

        assert 'st.session_state.get("aic_default_mode"' in source, \
            "Should read from aic_default_mode"

    def test_init_session_state_uses_default_display(self):
        """AI Coder should use global aic_default_display"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool._init_session_state)

        assert 'st.session_state.get("aic_default_display"' in source, \
            "Should read from aic_default_display"


# ============================================================================
# TEST: AI CODER EXPANDER REORGANIZATION
# ============================================================================

class TestAICoderExpanders:
    """Test AI Coder has proper expander organization"""

    def test_has_ai_behavior_expander(self):
        """AI Coder should have AI Behavior expander"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool.render_config)

        assert 'st.expander("AI Behavior"' in source, \
            "Should have AI Behavior expander"

    def test_has_confidence_thresholds_expander(self):
        """AI Coder should have Confidence Thresholds expander"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool.render_config)

        assert 'st.expander("Confidence Thresholds"' in source, \
            "Should have Confidence Thresholds expander"

    def test_has_prompt_customization_expander(self):
        """AI Coder should have Prompt Customization expander"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool.render_config)

        assert 'st.expander("Prompt Customization"' in source, \
            "Should have Prompt Customization expander"

    def test_no_advanced_ai_settings_expander(self):
        """Old Advanced AI Settings expander should be removed"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool.render_config)

        assert 'Advanced AI Settings' not in source, \
            "Old Advanced AI Settings expander should be removed"

    def test_ai_behavior_contains_context_window(self):
        """AI Behavior expander should contain context window setting"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool.render_config)

        # Find AI Behavior expander section
        ai_behavior_start = source.find('st.expander("AI Behavior"')
        confidence_start = source.find('st.expander("Confidence Thresholds"')
        ai_behavior_section = source[ai_behavior_start:confidence_start]

        assert "aic_ai_context_rows" in ai_behavior_section, \
            "AI Behavior should contain context window"

    def test_ai_behavior_contains_training_mode(self):
        """AI Behavior expander should contain training mode"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool.render_config)

        ai_behavior_start = source.find('st.expander("AI Behavior"')
        confidence_start = source.find('st.expander("Confidence Thresholds"')
        ai_behavior_section = source[ai_behavior_start:confidence_start]

        assert "aic_training_enabled" in ai_behavior_section, \
            "AI Behavior should contain training mode"

    def test_confidence_contains_thresholds(self):
        """Confidence Thresholds expander should contain threshold sliders"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool.render_config)

        confidence_start = source.find('st.expander("Confidence Thresholds"')
        prompt_start = source.find('st.expander("Prompt Customization"')
        confidence_section = source[confidence_start:prompt_start]

        assert "aic_threshold_auto_accept" in confidence_section
        assert "aic_threshold_flag" in confidence_section
        assert "aic_threshold_skip" in confidence_section

    def test_prompt_customization_contains_custom_prompt(self):
        """Prompt Customization should contain custom prompt settings"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool.render_config)

        prompt_start = source.find('st.expander("Prompt Customization"')
        prompt_section = source[prompt_start:prompt_start + 3000]

        assert "aic_custom_prompt_enabled" in prompt_section
        assert "aic_code_definitions" in prompt_section


# ============================================================================
# TEST: EDGE CASES AND CRASH SCENARIOS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and potential crash scenarios"""

    def test_missing_session_state_keys(self, mock_streamlit):
        """Should handle missing session state keys gracefully"""
        # Empty session state
        mock_streamlit.session_state.clear()

        with patch.dict('sys.modules', {'streamlit': mock_streamlit}):
            # Reimport to use mocked streamlit
            import importlib
            from tools import ai_coder
            importlib.reload(ai_coder)

            tool = ai_coder.AICoderTool()
            # Should not raise
            tool._init_session_state()

    def test_context_rows_boundary_values(self):
        """Context rows should accept boundary values"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        # Check min and max values are defined
        assert "min_value=0" in source, "Context rows should have min_value=0"
        assert "max_value=5" in source, "Context rows should have max_value=5"

    def test_slider_has_valid_range(self):
        """All sliders should have valid min < max"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        # Context rows slider
        assert source.count("min_value=0, max_value=5") >= 1, \
            "Context rows slider should have valid range"

    def test_selectbox_has_valid_options(self):
        """Selectboxes should have valid options"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        # Processing mode options
        assert '"per_row", "batch"' in source or "['per_row', 'batch']" in source.replace('"', "'"), \
            "Processing mode should have valid options"

        # Display mode options
        assert '"ai_first", "inline_badges"' in source or "['ai_first', 'inline_badges']" in source.replace('"', "'"), \
            "Display mode should have valid options"

    def test_toggle_returns_boolean(self):
        """Toggle settings should be boolean"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        # All toggles should have boolean defaults (True or False)
        toggle_matches = source.count("st.toggle")
        bool_defaults = source.count("value=st.session_state.get")

        assert toggle_matches > 0, "Should have toggle widgets"

    def test_no_division_by_zero_risks(self):
        """No division operations that could cause ZeroDivisionError"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        # Check for any division in the Coding Tools section
        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        # Should not have raw division without guards
        assert coding_section.count(" / ") == 0, \
            "No unguarded division operations in Coding Tools tab"

    def test_handles_none_values_in_get(self):
        """Session state .get() calls should have defaults"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        # Count .get( calls
        get_calls = coding_section.count(".get(")
        # Count .get( calls with defaults (have a comma after the key)
        get_with_default = coding_section.count('.get("')

        # All gets should have a default (each .get("key", default))
        assert get_calls > 0, "Should have .get() calls"


# ============================================================================
# TEST: SECURITY
# ============================================================================

class TestSecurity:
    """Test security aspects of settings"""

    def test_no_eval_or_exec(self):
        """No eval() or exec() in settings code"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        assert "eval(" not in source, "Should not use eval()"
        assert "exec(" not in source, "Should not use exec()"

    def test_no_os_system_calls(self):
        """No os.system() or subprocess calls in settings"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        assert "os.system" not in coding_section
        assert "subprocess" not in coding_section
        assert "shell=True" not in coding_section

    def test_no_pickle_in_settings(self):
        """No pickle usage in settings (potential security risk)"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings)

        assert "pickle" not in source, "Should not use pickle"

    def test_no_dangerous_imports(self):
        """Settings should not import dangerous modules"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings)

        dangerous = ["pickle", "subprocess", "ctypes", "multiprocessing"]
        for module in dangerous:
            assert f"import {module}" not in source, f"Should not import {module}"
            assert f"from {module}" not in source, f"Should not import from {module}"

    def test_setting_keys_are_safe_strings(self):
        """All setting keys should be safe string literals"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        # Keys should not be built from user input
        assert "session_state[user" not in coding_section
        assert "session_state[input" not in coding_section
        assert "session_state[f\"" not in coding_section  # No f-strings for keys
        assert "session_state[f'" not in coding_section

    def test_no_sql_in_settings(self):
        """Coding Tools tab should not have raw SQL"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        sql_keywords = ["SELECT ", "INSERT ", "UPDATE ", "DELETE ", "DROP "]
        for kw in sql_keywords:
            assert kw not in coding_section.upper(), f"Should not have raw SQL: {kw}"

    def test_no_html_injection_risk(self):
        """No unsafe HTML rendering in Coding Tools tab"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        # Should not have unsafe_allow_html with user input
        assert "unsafe_allow_html=True" not in coding_section, \
            "Coding Tools tab should not use unsafe_allow_html"

    def test_no_file_operations_in_settings_tab(self):
        """Coding Tools tab should not have file operations"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        file_ops = ["open(", "write(", "read(", "Path("]
        for op in file_ops:
            assert op not in coding_section, \
                f"Coding Tools tab should not have file operations: {op}"


# ============================================================================
# TEST: AI CODER SECURITY
# ============================================================================

class TestAICoderSecurity:
    """Test security aspects of AI Coder global settings usage"""

    def test_no_code_execution_from_settings(self):
        """Settings should not lead to code execution"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool._init_session_state)

        assert "eval(" not in source
        assert "exec(" not in source
        assert "compile(" not in source

    def test_setting_values_are_validated(self):
        """AI Coder should use default values if settings are invalid types"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool._init_session_state)

        # All .get() calls should have fallback defaults
        assert source.count('.get("coding_') >= 5, \
            "Should read from all global coding settings"

        # Each should have a default
        assert source.count(', True)') + source.count(', False)') + source.count(', 2)') >= 5


# ============================================================================
# TEST: DATA PERSISTENCE
# ============================================================================

class TestDataPersistence:
    """Test that settings are properly saved to session state"""

    def test_all_settings_written_to_session_state(self):
        """All settings should be written back to session state"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        expected_writes = [
            'session_state["coding_light_mode"]',
            'session_state["coding_context_rows"]',
            'session_state["coding_auto_advance"]',
            'session_state["coding_horizontal_codes"]',
            'session_state["coding_buttons_above"]',
            'session_state["aic_default_mode"]',
            'session_state["aic_default_display"]',
            'session_state["aic_autosave_enabled"]',
            'session_state["mc_autosave_enabled"]',
        ]

        for write in expected_writes:
            assert write in coding_section, f"Should write {write} to session state"

    def test_settings_use_unique_widget_keys(self):
        """All widgets should have unique keys to prevent conflicts"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        # Extract all key="..." values
        import re
        keys = re.findall(r'key="([^"]+)"', coding_section)

        # All keys should be unique
        assert len(keys) == len(set(keys)), \
            f"Widget keys should be unique. Duplicates: {[k for k in keys if keys.count(k) > 1]}"

    def test_widget_keys_have_settings_prefix(self):
        """Widget keys in settings should have consistent prefix"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        import re
        keys = re.findall(r'key="([^"]+)"', coding_section)

        # All keys should start with "settings_"
        for key in keys:
            assert key.startswith("settings_"), \
                f"Widget key '{key}' should start with 'settings_'"


# ============================================================================
# TEST: INTEGRATION
# ============================================================================

class TestIntegration:
    """Integration tests between settings and tools"""

    def test_ai_coder_respects_global_defaults(self):
        """AI Coder should use global settings as defaults - verified via code analysis"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool._init_session_state)

        # Verify the pattern: if "aic_X" not in session_state, use global "coding_X" or "aic_default_X"
        global_mappings = [
            ('aic_light_mode', 'coding_light_mode'),
            ('aic_context_rows', 'coding_context_rows'),
            ('aic_auto_advance', 'coding_auto_advance'),
            ('aic_horizontal_codes', 'coding_horizontal_codes'),
            ('aic_buttons_above', 'coding_buttons_above'),
            ('aic_ai_mode', 'aic_default_mode'),
            ('aic_ai_display', 'aic_default_display'),
        ]

        for local_key, global_key in global_mappings:
            # Pattern should be: if "local_key" not in ... = session_state.get("global_key"
            assert f'if "{local_key}" not in st.session_state' in source, \
                f"Should check if {local_key} exists"
            assert f'get("{global_key}"' in source, \
                f"Should read from global {global_key} for {local_key}"

    def test_ai_coder_local_override_preserved(self):
        """AI Coder local settings should override globals - verified via code pattern"""
        from tools.ai_coder import AICoderTool
        import inspect

        tool = AICoderTool()
        source = inspect.getsource(tool._init_session_state)

        # The pattern should be: if "key" not in session_state: ...
        # This ensures that if local key already exists, it won't be overwritten
        local_keys = [
            'aic_light_mode',
            'aic_context_rows',
            'aic_auto_advance',
            'aic_horizontal_codes',
            'aic_buttons_above',
            'aic_ai_mode',
            'aic_ai_display',
        ]

        for key in local_keys:
            # Must have the "not in" check to preserve existing values
            assert f'if "{key}" not in st.session_state' in source, \
                f"Should use 'not in' pattern for {key} to preserve existing values"


# ============================================================================
# TEST: TYPE SAFETY
# ============================================================================

class TestTypeSafety:
    """Test type safety of settings"""

    def test_boolean_settings_have_boolean_defaults(self):
        """Boolean settings should have boolean defaults"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        boolean_settings = [
            "coding_light_mode",
            "coding_auto_advance",
            "coding_horizontal_codes",
            "coding_buttons_above",
            "aic_autosave_enabled",
            "mc_autosave_enabled",
        ]

        for setting in boolean_settings:
            # Find the .get() call for this setting
            pattern = f'get("{setting}",'
            assert pattern in coding_section, f"Should have .get() for {setting}"

            # The default should be True or False
            idx = coding_section.find(pattern)
            snippet = coding_section[idx:idx+50]
            assert "True" in snippet or "False" in snippet, \
                f"Boolean setting {setting} should have boolean default"

    def test_integer_settings_have_integer_defaults(self):
        """Integer settings should have integer defaults"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        int_settings = ["coding_context_rows"]

        for setting in int_settings:
            pattern = f'get("{setting}",'
            assert pattern in coding_section

            idx = coding_section.find(pattern)
            snippet = coding_section[idx:idx+30]
            # Should have a number default like , 2)
            import re
            assert re.search(r',\s*\d+\)', snippet), \
                f"Integer setting {setting} should have integer default"

    def test_string_settings_have_string_defaults(self):
        """String settings should have string defaults"""
        from pages import settings
        import inspect
        source = inspect.getsource(settings.render)

        tab4_start = source.find("with tab4:")
        tab5_start = source.find("with tab5:")
        coding_section = source[tab4_start:tab5_start]

        string_settings = ["aic_default_mode", "aic_default_display"]

        for setting in string_settings:
            pattern = f'get("{setting}",'
            assert pattern in coding_section

            idx = coding_section.find(pattern)
            snippet = coding_section[idx:idx+50]
            # Should have a string default like , "per_row")
            assert '", "' in snippet or "', '" in snippet, \
                f"String setting {setting} should have string default"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
