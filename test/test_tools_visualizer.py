"""
Tests for ToolsVisualizer

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from elastic_gumby_universal_orch_agent_prototype.visualizer.tools_visualizer import ToolsVisualizer


class TestVisualizeToolsEmptyInput:
    """Test visualize_tools method with empty input."""

    def test_visualize_empty_tools_list(self):
        """Test visualization with empty tools list."""
        visualizer = ToolsVisualizer()
        result = visualizer.visualize_tools([])

        # Should contain header and no tools message
        assert "PROCESSED TOOL DEFINITIONS" in result
        assert "No processed tools available" in result
        assert "=" * 80 in result

    def test_visualize_empty_tools_no_colors(self):
        """Test visualization with empty tools list and no colors."""
        visualizer = ToolsVisualizer(use_colors=False)
        result = visualizer.visualize_tools([])

        # Should not contain ANSI color codes
        assert "\033[" not in result
        assert "PROCESSED TOOL DEFINITIONS" in result
        assert "No processed tools available" in result

    def test_visualize_empty_tools_no_icons(self):
        """Test visualization with empty tools list and no icons."""
        visualizer = ToolsVisualizer(use_icons=False)
        result = visualizer.visualize_tools([])

        # Should not contain Unicode icons when use_icons=False
        assert "🔧" not in result
        assert "📭" not in result
        assert "PROCESSED TOOL DEFINITIONS" in result
        assert "No processed tools available" in result


class TestVisualizeToolsBasicTool:
    """Test visualize_tools method with basic tool definitions."""

    def test_visualize_multiple_basic_tools(self):
        """Test visualization with multiple basic tools."""
        tools = [
            {
                "name": "tool_one",
                "description": "First tool",
                "resource": "resource1",
                "parameters": [],
                "return": {},
            },
            {
                "name": "tool_two",
                "description": "Second tool",
                "resource": "resource2",
                "parameters": [],
                "return": {},
            },
        ]

        visualizer = ToolsVisualizer()
        result = visualizer.visualize_tools(tools)

        assert "tool_one" in result
        assert "tool_two" in result
        assert "First tool" in result
        assert "Second tool" in result
        assert "├──" in result  # First tool indicator
        assert "└──" in result  # Last tool indicator

    def test_visualize_tool_missing_optional_fields(self):
        """Test visualization with tool missing optional fields."""
        tools = [
            {
                "name": "minimal_tool",
                "description": "Minimal tool definition",
                # Missing resource, parameters, returns
            }
        ]

        visualizer = ToolsVisualizer()
        result = visualizer.visualize_tools(tools)

        assert "minimal_tool" in result
        assert "Minimal tool definition" in result
        assert "Resource: N/A" in result
        assert "No parameters" in result


class TestVisualizeToolsWithParameters:
    """Test visualize_tools method with tools containing parameters."""

    def test_visualize_tool_with_required_parameter(self):
        """Test visualization with tool having required parameters."""
        tools = [
            {
                "name": "param_tool",
                "description": "Tool with parameters",
                "parameters": [
                    {
                        "name": "input_text",
                        "type": "string",
                        "description": "Text to process",
                        "required": True,
                    }
                ],
            }
        ]

        visualizer = ToolsVisualizer()
        result = visualizer.visualize_tools(tools)

        assert "param_tool" in result
        assert "Parameters (1)" in result
        assert "input_text" in result
        assert "Type: string" in result
        assert "Text to process" in result
        assert "●" in result  # Required indicator

    def test_visualize_tool_with_optional_parameter(self):
        """Test visualization with tool having optional parameters."""
        tools = [
            {
                "name": "optional_tool",
                "description": "Tool with optional parameter",
                "parameters": [
                    {
                        "name": "optional_param",
                        "type": "boolean",
                        "description": "Optional parameter",
                        "required": False,
                        "default_value": True,
                    }
                ],
            }
        ]

        visualizer = ToolsVisualizer()
        result = visualizer.visualize_tools(tools)

        assert "optional_param" in result
        assert "Type: boolean" in result
        assert "Optional parameter" in result
        assert "Default: True" in result
        assert "○" in result  # Optional indicator

    def test_visualize_tool_with_parameter_constraints(self):
        """Test visualization with parameter constraints."""
        tools = [
            {
                "name": "constrained_tool",
                "description": "Tool with constrained parameters",
                "parameters": [
                    {
                        "name": "number_param",
                        "type": "number",
                        "description": "A number with constraints",
                        "required": True,
                        "constraints": {"min": 1, "max": 100, "pattern": "^[0-9]+$"},
                    },
                    {
                        "name": "enum_param",
                        "type": "string",
                        "description": "Enum parameter",
                        "required": False,
                        "constraints": {"enum": ["option1", "option2", "option3"]},
                    },
                ],
            }
        ]

        visualizer = ToolsVisualizer()
        result = visualizer.visualize_tools(tools)

        assert "number_param" in result
        assert "enum_param" in result
        assert "min: 1" in result
        assert "max: 100" in result
        assert "pattern: ^[0-9]+$" in result
        assert "enum: ['option1', 'option2', 'option3']" in result

    def test_visualize_tool_with_multiple_parameters(self):
        """Test visualization with multiple parameters."""
        tools = [
            {
                "name": "multi_param_tool",
                "description": "Tool with multiple parameters",
                "parameters": [
                    {
                        "name": "first_param",
                        "type": "string",
                        "description": "First parameter",
                        "required": True,
                    },
                    {
                        "name": "second_param",
                        "type": "number",
                        "description": "Second parameter",
                        "required": False,
                    },
                    {
                        "name": "third_param",
                        "type": "array",
                        "description": "Third parameter",
                        "required": True,
                    },
                ],
            }
        ]

        visualizer = ToolsVisualizer()
        result = visualizer.visualize_tools(tools)

        assert "Parameters (3)" in result
        assert "first_param" in result
        assert "second_param" in result
        assert "third_param" in result
        # Check for proper tree structure
        assert result.count("├──") >= 2  # At least 2 non-last parameters
        assert result.count("└──") >= 1  # At least 1 last parameter


class TestVisualizeToolsWithReturns:
    """Test visualize_tools method with tools containing return specifications."""

    def test_visualize_tool_with_returns(self):
        """Test visualization with tool having return specification."""
        tools = [
            {
                "name": "return_tool",
                "description": "Tool with return specification",
                "parameters": [],
                "return": {
                    "type": "object",
                    "description": "The processed result",
                },
            }
        ]

        visualizer = ToolsVisualizer()
        result = visualizer.visualize_tools(tools)

        assert "return_tool" in result
        assert "Return:" in result
        assert "Type: object" in result
        assert "The processed result" in result

    def test_visualize_tool_with_returns_no_name(self):
        """Test visualization with return specification missing name."""
        tools = [
            {
                "name": "unnamed_return_tool",
                "description": "Tool with unnamed return",
                "parameters": [],
                "return": {"type": "string", "description": "A string result"},
            }
        ]

        visualizer = ToolsVisualizer()
        result = visualizer.visualize_tools(tools)

        assert "Return:" in result
        assert "Type: string" in result
        assert "A string result" in result

    def test_visualize_tool_with_returns_minimal(self):
        """Test visualization with minimal return specification."""
        tools = [
            {
                "name": "minimal_return_tool",
                "description": "Tool with minimal return",
                "parameters": [],
                "return": {"type": "boolean"},
            }
        ]

        visualizer = ToolsVisualizer()
        result = visualizer.visualize_tools(tools)

        assert "Return:" in result
        assert "Type: boolean" in result
        assert "No description" in result


class TestVisualizeToolsComplexScenarios:
    """Test visualize_tools method with complex tool definitions."""

    def test_visualize_multiple_complex_tools(self):
        """Test visualization with multiple complex tools."""
        tools = [
            {
                "name": "tool_alpha",
                "description": "First complex tool",
                "resource": "alpha",
                "parameters": [
                    {
                        "name": "param1",
                        "type": "string",
                        "description": "Parameter 1",
                        "required": True,
                    }
                ],
                "return": {"type": "string", "description": "String result"},
            },
            {
                "name": "tool_beta",
                "description": "Second complex tool",
                "resource": "beta",
                "parameters": [
                    {
                        "name": "param2",
                        "type": "number",
                        "description": "Parameter 2",
                        "required": False,
                    }
                ],
                "return": {"type": "boolean", "description": "Boolean result"},
            },
        ]

        visualizer = ToolsVisualizer()
        result = visualizer.visualize_tools(tools)

        # Verify proper tree structure for multiple tools
        assert "tool_alpha" in result
        assert "tool_beta" in result
        assert result.count("├──") >= 1  # First tool
        assert result.count("└──") >= 1  # Last tool


class TestVisualizeToolsConstraints:
    """Test constraint handling in visualize_tools method."""

    def test_build_constraint_parts_all_types(self):
        """Test _build_constraint_parts method with all constraint types."""
        visualizer = ToolsVisualizer()

        constraints = {"min": 1, "max": 100, "pattern": "^test$", "enum": ["a", "b", "c"]}

        parts = visualizer._build_constraint_parts(constraints)

        assert "min: 1" in parts
        assert "max: 100" in parts
        assert "pattern: ^test$" in parts
        assert "enum: ['a', 'b', 'c']" in parts
        assert len(parts) == 4

    def test_build_constraint_parts_partial(self):
        """Test _build_constraint_parts method with partial constraints."""
        visualizer = ToolsVisualizer()

        constraints = {"min": 5, "pattern": "test"}
        parts = visualizer._build_constraint_parts(constraints)

        assert "min: 5" in parts
        assert "pattern: test" in parts
        assert len(parts) == 2

    def test_build_constraint_parts_empty(self):
        """Test _build_constraint_parts method with empty constraints."""
        visualizer = ToolsVisualizer()

        parts = visualizer._build_constraint_parts({})
        assert len(parts) == 0


class TestSaveToolsVisualization:
    """Test save_tools_visualization method."""

    def test_save_tools_visualization_strips_ansi(self):
        """Test that saved visualization strips ANSI codes."""
        tools = [
            {
                "name": "ansi_test_tool",
                "description": "Tool for testing ANSI stripping",
                "parameters": [],
            }
        ]

        visualizer = ToolsVisualizer(use_colors=True)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as temp_file:
            temp_path = temp_file.name

        try:
            with patch("builtins.print"):
                visualizer.save_tools_visualization(tools, temp_path)

            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Should not contain ANSI escape codes
            assert "\033[" not in content
            assert "ansi_test_tool" in content

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_save_tools_visualization_file_error(self, mock_open):
        """Test save_tools_visualization with file write error."""
        tools = [{"name": "test_tool", "description": "Test"}]
        visualizer = ToolsVisualizer()

        # Should raise the IOError
        with pytest.raises(IOError, match="Permission denied"):
            visualizer.save_tools_visualization(tools, "/invalid/path/file.md")


class TestVisualizeToolsEdgeCases:
    """Test edge cases and error conditions."""

    def test_visualize_tools_none_input(self):
        """Test visualization with None input."""
        visualizer = ToolsVisualizer()

        # The implementation handles None gracefully by treating it as falsy (empty)
        result = visualizer.visualize_tools(None)

        # Should behave like empty list
        assert "PROCESSED TOOL DEFINITIONS" in result
        assert "No processed tools available" in result

    def test_visualize_tools_malformed_tool(self):
        """Test visualization with malformed tool definition."""
        tools = [
            {
                # Missing required fields
                "description": "Tool without name"
            }
        ]

        visualizer = ToolsVisualizer()

        # Should handle missing name gracefully
        with pytest.raises(KeyError):
            visualizer.visualize_tools(tools)

    def test_visualize_tools_parameter_missing_fields(self):
        """Test visualization with parameter missing required fields."""
        tools = [
            {
                "name": "incomplete_param_tool",
                "description": "Tool with incomplete parameter",
                "parameters": [
                    {
                        # Missing name, type, description
                        "required": True
                    }
                ],
            }
        ]

        visualizer = ToolsVisualizer()

        # Should raise KeyError when parameter name is missing
        with pytest.raises(KeyError):
            visualizer.visualize_tools(tools)

    def test_visualize_tools_with_none_values(self):
        """Test visualization with None values in tool definition."""
        tools = [
            {
                "name": "none_values_tool",
                "description": None,
                "resource": None,
                "parameters": None,
                "return": None,
            }
        ]

        visualizer = ToolsVisualizer()

        # Should raise TypeError when trying to concatenate None with string
        with pytest.raises(TypeError):
            visualizer.visualize_tools(tools)
