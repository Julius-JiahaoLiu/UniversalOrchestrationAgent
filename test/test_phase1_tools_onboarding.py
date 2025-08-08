"""
Tests for Phase1ToolsOnboarding

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

from unittest.mock import Mock, patch

import pytest

from elastic_gumby_universal_orch_agent_prototype.phases.phase1_tools_onboarding import Phase1ToolsOnboarding

class TestCollectToolDescriptions:
    """Test collect_tool_descriptions method."""

    @pytest.fixture
    def mock_session_data(self):
        """Create mock session data."""
        return {"tools": []}

    @pytest.fixture
    def mock_get_user_input(self):
        """Create mock get_user_input function."""
        return Mock()

    @pytest.fixture
    def mock_tools_visualizer(self):
        """Create mock tools visualizer."""
        return Mock()

    @pytest.fixture
    def phase1_instance(self, mock_session_data, mock_get_user_input, mock_tools_visualizer):
        """Create Phase1ToolsOnboarding instance with mocked dependencies."""
        return Phase1ToolsOnboarding(
            session_data=mock_session_data,
            get_user_input_func=mock_get_user_input,
            tools_visualizer=mock_tools_visualizer
        )

    def test_collect_tool_descriptions_quit_immediately(self, phase1_instance, mock_get_user_input):
        """Test collect_tool_descriptions when user quits immediately."""
        mock_get_user_input.return_value = "quit"

        result = phase1_instance.collect_tool_descriptions()

        assert result is False
        mock_get_user_input.assert_called_once()

    def test_collect_tool_descriptions_done_immediately(self, phase1_instance, mock_get_user_input):
        """Test collect_tool_descriptions when user types 'done' immediately."""
        mock_get_user_input.return_value = "done"

        with patch('builtins.print'):
            result = phase1_instance.collect_tool_descriptions()

        assert result is True
        mock_get_user_input.assert_called_once()

    def test_collect_tool_descriptions_empty_input_then_done(self, phase1_instance, mock_get_user_input):
        """Test collect_tool_descriptions with empty input followed by 'done'."""
        mock_get_user_input.side_effect = ["", "done"]

        with patch('builtins.print'):
            result = phase1_instance.collect_tool_descriptions()

        assert result is True
        assert mock_get_user_input.call_count == 2

    def test_collect_tool_descriptions_successful_json_load(self, phase1_instance, mock_get_user_input):
        """Test collect_tool_descriptions with successful JSON tool loading."""
        mock_tools_loader = Mock()
        mock_tools_loader.load_tools_from_json_string.return_value = {
            "success": True,
            "tools": [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": [],
                    "returns": {"name": "result", "type": "string", "description": "Test result"}
                }
            ]
        }
        phase1_instance.tools_transformer.tools_loader = mock_tools_loader

        mock_get_user_input.side_effect = ['{"name": "test_tool"}', "done"]

        with patch('builtins.print'):
            result = phase1_instance.collect_tool_descriptions()

        assert result is True
        assert len(phase1_instance.session_data["tools"]) == 1
        assert phase1_instance.session_data["tools"][0]["name"] == "test_tool"
        mock_tools_loader.load_tools_from_json_string.assert_called_once_with('{"name": "test_tool"}')

    def test_collect_tool_descriptions_failed_json_load(self, phase1_instance, mock_get_user_input):
        """Test collect_tool_descriptions with failed JSON tool loading."""
        mock_tools_loader = Mock()
        mock_tools_loader.load_tools_from_json_string.return_value = {
            "success": False,
            "message": "Invalid JSON format"
        }
        phase1_instance.tools_transformer.tools_loader = mock_tools_loader

        # Mock the transform_description method to return None (failed transformation)
        mock_transform_description = Mock(return_value=None)
        phase1_instance.tools_transformer.transform_description = mock_transform_description

        mock_get_user_input.side_effect = ['invalid json', "done"]

        with patch('builtins.print'):
            result = phase1_instance.collect_tool_descriptions()

        assert result is True
        assert len(phase1_instance.session_data["tools"]) == 0
        mock_tools_loader.load_tools_from_json_string.assert_called_once_with('invalid json')
        mock_transform_description.assert_called_once_with('invalid json')

    def test_collect_tool_descriptions_successful_transformation_after_failed_json(self, phase1_instance, mock_get_user_input):
        """Test collect_tool_descriptions with successful transformation after failed JSON loading."""
        mock_tools_loader = Mock()
        mock_tools_loader.load_tools_from_json_string.return_value = {
            "success": False,
            "message": "Invalid JSON format"
        }
        phase1_instance.tools_transformer.tools_loader = mock_tools_loader

        # Mock the transform_description method to return a valid tool
        transformed_tool = {
            "name": "transformed_tool",
            "description": "A tool created via transformation",
            "parameters": [],
            "returns": {"name": "result", "type": "string", "description": "Transformed result"}
        }
        mock_transform_description = Mock(return_value=transformed_tool)
        phase1_instance.tools_transformer.transform_description = mock_transform_description

        mock_get_user_input.side_effect = ['raw tool description', "done"]

        with patch('builtins.print'):
            result = phase1_instance.collect_tool_descriptions()

        assert result is True
        assert len(phase1_instance.session_data["tools"]) == 1
        assert phase1_instance.session_data["tools"][0]["name"] == "transformed_tool"
        mock_tools_loader.load_tools_from_json_string.assert_called_once_with('raw tool description')
        mock_transform_description.assert_called_once_with('raw tool description')

    def test_collect_tool_descriptions_multiple_tools_then_done(self, phase1_instance, mock_get_user_input):
        """Test collect_tool_descriptions with multiple tool inputs."""
        mock_tools_loader = Mock()
        mock_tools_loader.load_tools_from_json_string.side_effect = [
            {
                "success": True,
                "tools": [{"name": "tool1", "description": "First tool"}]
            },
            {
                "success": True,
                "tools": [{"name": "tool2", "description": "Second tool"}]
            }
        ]
        phase1_instance.tools_transformer.tools_loader = mock_tools_loader

        mock_get_user_input.side_effect = ['tool1_json', 'tool2_json', "done"]

        with patch('builtins.print'):
            result = phase1_instance.collect_tool_descriptions()

        assert result is True
        assert len(phase1_instance.session_data["tools"]) == 2
        assert mock_tools_loader.load_tools_from_json_string.call_count == 2


class TestHandlePostProcessingOptions:
    """Test handle_post_processing_options method."""

    @pytest.fixture
    def mock_session_data(self):
        """Create mock session data with sample tools."""
        return {
            "tools": [
                {
                    "name": "sample_tool",
                    "description": "A sample tool for testing",
                    "parameters": [],
                    "returns": {"name": "result", "type": "string", "description": "Sample result"}
                }
            ]
        }

    @pytest.fixture
    def mock_get_user_input(self):
        """Create mock get_user_input function."""
        return Mock()

    @pytest.fixture
    def mock_tools_visualizer(self):
        """Create mock tools visualizer."""
        mock_visualizer = Mock()
        mock_visualizer.visualize_tools.return_value = "Mocked visualization output"
        return mock_visualizer

    @pytest.fixture
    def phase1_instance(self, mock_session_data, mock_get_user_input, mock_tools_visualizer):
        """Create Phase1ToolsOnboarding instance with mocked dependencies."""
        return Phase1ToolsOnboarding(
            session_data=mock_session_data,
            get_user_input_func=mock_get_user_input,
            tools_visualizer=mock_tools_visualizer
        )

    def test_handle_post_processing_options_choice_1_review_tools(self, phase1_instance, mock_get_user_input, mock_tools_visualizer):
        """Test handle_post_processing_options with choice 1 (review tools)."""
        # First call returns "1" to review tools, second call returns "3" to proceed
        mock_get_user_input.side_effect = ["1", "3"]

        with patch('builtins.print'):
            result = phase1_instance.handle_post_processing_options()

        assert result is True
        assert mock_get_user_input.call_count == 2
        mock_tools_visualizer.visualize_tools.assert_called_once_with(phase1_instance.session_data["tools"])

    def test_handle_post_processing_options_choice_2_add_more_tools(self, phase1_instance, mock_get_user_input):
        """Test handle_post_processing_options with choice 2 (add more tools)."""
        # Mock collect_tool_descriptions to return True
        with patch.object(phase1_instance, 'collect_tool_descriptions', return_value=True):
            mock_get_user_input.side_effect = ["2", "3"]

            with patch('builtins.print'):
                result = phase1_instance.handle_post_processing_options()

            assert result is True
            assert mock_get_user_input.call_count == 2

    def test_handle_post_processing_options_choice_3_proceed(self, phase1_instance, mock_get_user_input):
        """Test handle_post_processing_options with choice 3 (proceed to Phase 2)."""
        mock_get_user_input.return_value = "3"

        with patch('builtins.print'):
            result = phase1_instance.handle_post_processing_options()

        assert result is True
        mock_get_user_input.assert_called_once()


class TestPhase1PrintMethods:
    """Test print methods for Phase1ToolsOnboarding."""

    @pytest.fixture
    def phase1_instance(self):
        """Create Phase1ToolsOnboarding instance for print method testing."""
        return Phase1ToolsOnboarding(
            session_data={"tools": []},
            get_user_input_func=Mock(),
            tools_visualizer=Mock()
        )

    def test_print_phase_header(self, phase1_instance):
        """Test print_phase_header method."""
        with patch('builtins.print') as mock_print:
            phase1_instance.print_phase_header()

        # Verify that print was called multiple times
        assert mock_print.call_count > 0
        # Check that the header contains expected text
        call_args = [str(call) for call in mock_print.call_args_list]
        header_found = any("PHASE 1: AVAILABLE TOOLS ONBOARDING" in arg for arg in call_args)
        assert header_found

    def test_print_detailed_guidance(self, phase1_instance):
        """Test print_detailed_guidance method."""
        with patch('builtins.print') as mock_print:
            phase1_instance.print_detailed_guidance()

        # Verify that print was called multiple times for the detailed guidance
        assert mock_print.call_count > 10
        call_args = [str(call) for call in mock_print.call_args_list]
        guidance_found = any("Required Information for Each Tool" in arg for arg in call_args)
        assert guidance_found
