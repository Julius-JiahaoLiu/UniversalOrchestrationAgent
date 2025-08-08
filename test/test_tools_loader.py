"""
Tests for ToolsLoader

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

import json
from unittest.mock import patch

import pytest

from elastic_gumby_universal_orch_agent_prototype.visualizer.base import Colors
from elastic_gumby_universal_orch_agent_prototype.visualizer.tools_loader import ToolsLoader


class TestLoadToolsFromJsonString:
    """Test load_tools_from_json_string method."""

    def test_load_valid_tools_list(self):
        """Test loading valid tools from JSON list."""
        tools_json = json.dumps(
            [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "resource": "arn:aws:lambda:us-west-2:123456789012:function:test_function",
                    "parameters": [
                        {"name": "param1", "type": "string", "description": "Test parameter"}
                    ],
                    "return": {"type": "string", "description": "Test result"},
                }
            ]
        )

        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status") as mock_print:
            result = loader.load_tools_from_json_string(tools_json)

        assert result["success"] is True
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "test_tool"

        # Verify success message was printed
        mock_print.assert_any_call("Successfully loaded 1 tool(s) from JSON", Colors.GREEN)

    def test_load_tools_from_dict_with_available_tools_key(self):
        """Test loading tools from dictionary with 'available_tools' key."""
        tools_data = {
            "available_tools": [
                {
                    "name": "available_tool",
                    "description": "Available tool",
                    "resource": "arn:aws:lambda:us-west-2:123456789012:function:available_function",
                    "parameters": [],
                    "return": {"type": "string", "description": "String result"},
                }
            ]
        }

        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status"):
            result = loader.load_tools_from_json_string(json.dumps(tools_data))

        assert result["success"] is True
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "available_tool"

    def test_load_single_tool_from_dict(self):
        """Test loading single tool from dictionary."""
        single_tool = {
            "name": "single_tool",
            "description": "A single tool",
            "resource": "arn:aws:lambda:us-west-2:123456789012:function:single_function",
            "parameters": [],
            "return": {"type": "number", "description": "Numeric result"},
        }

        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status"):
            result = loader.load_tools_from_json_string(json.dumps(single_tool))

        assert result["success"] is True
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "single_tool"

    def test_load_invalid_json(self):
        """Test loading invalid JSON string."""
        invalid_json = '{"invalid": json syntax'

        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status") as mock_print:
            result = loader.load_tools_from_json_string(invalid_json)

        assert result["success"] is False
        assert result["tools"] == []

        # Verify error message was printed
        mock_print.assert_called()
        error_call = [
            call for call in mock_print.call_args_list if "Invalid JSON format" in str(call)
        ]
        assert len(error_call) > 0

    def test_load_empty_json(self):
        """Test loading empty JSON."""
        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status") as mock_print:
            result = loader.load_tools_from_json_string("{}")

        assert result["success"] is False
        assert result["tools"] == []

    def test_load_json_with_validation_failure(self):
        """Test loading JSON that fails validation."""
        invalid_tools = [
            {
                "name": "invalid_tool",
                # Missing required fields
            }
        ]

        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status"):
            result = loader.load_tools_from_json_string(json.dumps(invalid_tools))

        assert result["success"] is False
        assert result["tools"] == []


class TestExtractToolsFromData:
    """Test _extract_tools_from_data method."""

    def test_extract_from_list(self):
        """Test extracting tools from direct list."""
        loader = ToolsLoader()
        tools_list = [{"name": "tool1"}, {"name": "tool2"}]

        result = loader._extract_tools_from_data(tools_list)

        assert result == tools_list
        assert len(result) == 2

    def test_extract_from_dict_with_available_tools(self):
        """Test extracting tools from dict with 'available_tools' key."""
        loader = ToolsLoader()
        data = {"available_tools": [{"name": "tool1"}], "other_data": "ignored"}

        result = loader._extract_tools_from_data(data)

        assert result == [{"name": "tool1"}]

    def test_extract_from_dict_with_tools(self):
        """Test extracting tools from dict with 'tools' key."""
        loader = ToolsLoader()
        data = {"tools": [{"name": "tool1"}], "metadata": "ignored"}

        result = loader._extract_tools_from_data(data)

        assert result == [{"name": "tool1"}]

    def test_extract_from_dict_with_tool_definitions(self):
        """Test extracting tools from dict with 'tool_definitions' key."""
        loader = ToolsLoader()
        data = {"tool_definitions": [{"name": "tool1"}], "version": "1.0"}

        result = loader._extract_tools_from_data(data)

        assert result == [{"name": "tool1"}]

    def test_extract_from_invalid_data_type(self):
        """Test extracting from invalid data type."""
        loader = ToolsLoader()

        result = loader._extract_tools_from_data("invalid_data")

        assert result == []


class TestValidateToolsCollection:
    """Test _validate_tools_collection method."""

    def test_validate_empty_tools_list(self):
        """Test validation of empty tools list."""
        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status") as mock_print:
            result = loader._validate_tools_collection([])

        assert result is False
        mock_print.assert_called_with("✘ No tools provided for validation", Colors.RED)

    def test_validate_valid_tools_collection(self):
        """Test validation of valid tools collection."""
        tools = [
            {
                "name": "valid_tool",
                "description": "A valid tool",
                "resource": "arn:aws:lambda:us-west-2:123456789012:function:valid_function",
                "parameters": [{"name": "param1", "type": "string", "description": "Parameter 1"}],
                "return": {"type": "string", "description": "Result"},
            }
        ]

        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status") as mock_print:
            result = loader._validate_tools_collection(tools)

        assert result is True
        # Should print success message for valid tool
        success_calls = [call for call in mock_print.call_args_list if "is valid" in str(call)]
        assert len(success_calls) > 0

    def test_validate_mixed_tools_collection(self):
        """Test validation of collection with both valid and invalid tools."""
        tools = [
            {
                "name": "valid_tool",
                "description": "A valid tool",
                "resource": "arn:aws:lambda:us-west-2:123456789012:function:valid_function",
                "parameters": [],
                "return": {"type": "string", "description": "Result"},
            },
            {
                "name": "invalid_tool",
                # Missing required fields
            },
        ]

        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status") as mock_print:
            result = loader._validate_tools_collection(tools)

        assert result is False
        # Should print both success and error messages
        success_calls = [call for call in mock_print.call_args_list if "is valid" in str(call)]
        error_calls = [call for call in mock_print.call_args_list if "has issues" in str(call)]
        assert len(success_calls) > 0
        assert len(error_calls) > 0

    def test_validate_all_invalid_tools_collection(self):
        """Test validation of collection with all invalid tools."""
        tools = [
            {"name": "tool1"},  # Missing required fields
            {"description": "tool2"},  # Missing required fields
        ]

        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status") as mock_print:
            result = loader._validate_tools_collection(tools)

        assert result is False
        # Should print error messages for all tools
        error_calls = [call for call in mock_print.call_args_list if "has issues" in str(call)]
        assert len(error_calls) == 2


class TestValidateToolStructure:
    """Test _validate_tool_structure method."""

    def test_validate_tool_missing_required_fields(self):
        """Test validation of tool missing required fields."""
        tool = {
            "name": "incomplete_tool",
            # Missing description, resource, parameters, returns
        }

        loader = ToolsLoader()
        result = loader._validate_tool_structure(tool)

        assert result["valid"] is False
        assert len(result["errors"]) == 4  # Missing 4 required fields
        assert any("Missing required field: description" in error for error in result["errors"])
        assert any("Missing required field: resource" in error for error in result["errors"])
        assert any("Missing required field: parameters" in error for error in result["errors"])
        assert any("Missing required field: return" in error for error in result["errors"])

    def test_validate_tool_with_invalid_parameters_type(self):
        """Test validation of tool with invalid parameters type."""
        tool = {
            "name": "invalid_params_tool",
            "description": "Tool with invalid params",
            "resource": "arn:aws:lambda:us-west-2:123456789012:function:my_function",
            "parameters": "not_a_list",  # Should be list
            "returns": {"name": "result", "type": "string", "description": "Result"},
        }

        loader = ToolsLoader()
        result = loader._validate_tool_structure(tool)

        assert result["valid"] is False
        assert any("Parameters should be a list" in error for error in result["errors"])

    def test_validate_tool_with_invalid_parameter_structure(self):
        """Test validation of tool with invalid parameter structure."""
        tool = {
            "name": "invalid_param_structure_tool",
            "description": "Tool with invalid param structure",
            "resource": "arn:aws:lambda:us-west-2:123456789012:function:my_function",
            "parameters": [
                {"name": "valid_param", "type": "string", "description": "Valid parameter"},
                {
                    "name": "invalid_param",
                    # Missing type and description
                },
            ],
            "returns": {"name": "result", "type": "string", "description": "Result"},
        }

        loader = ToolsLoader()
        result = loader._validate_tool_structure(tool)

        assert result["valid"] is False
        assert any(
            "Parameter at index 1 missing required field: type" in error
            for error in result["errors"]
        )
        # Note: validation stops at first missing field, so description error may not appear

    def test_validate_tool_with_invalid_returns_structure(self):
        """Test validation of tool with invalid returns structure."""
        tool = {
            "name": "invalid_returns_tool",
            "description": "Tool with invalid returns",
            "resource": "arn:aws:lambda:us-west-2:123456789012:function:my_function",
            "parameters": [],
            "return": {
                # Missing type and description
            },
        }

        loader = ToolsLoader()
        result = loader._validate_tool_structure(tool)

        assert result["valid"] is False
        assert any(
            "Return at index 0 missing required field: type" in error
            for error in result["errors"]
        )
        # Note: validation stops at first missing field, so description error may not appear


class TestValidateParameter:
    """Test _validate_parameter method."""

    def test_validate_parameter_not_dict(self):
        """Test validation of parameter that is not a dictionary."""
        loader = ToolsLoader()

        with pytest.raises(ValueError, match="Parameter at index 0 should be a dictionary"):
            loader._validate_parameter("not_a_dict", 0, "Parameter")

    def test_validate_parameter_missing_name(self):
        """Test validation of parameter missing name field."""
        param = {"type": "string", "description": "Missing name"}

        loader = ToolsLoader()

        with pytest.raises(ValueError, match="Parameter at index 0 missing required field: name"):
            loader._validate_parameter(param, 0, "Parameter")

    def test_validate_parameter_missing_type(self):
        """Test validation of parameter missing type field."""
        param = {"name": "param_name", "description": "Missing type"}

        loader = ToolsLoader()

        with pytest.raises(ValueError, match="Parameter at index 0 missing required field: type"):
            loader._validate_parameter(param, 0, "Parameter")

    def test_validate_parameter_missing_description(self):
        """Test validation of parameter missing description field."""
        param = {"name": "param_name", "type": "string"}

        loader = ToolsLoader()

        with pytest.raises(
            ValueError, match="Parameter at index 0 missing required field: description"
        ):
            loader._validate_parameter(param, 0, "Parameter")

    def test_validate_return_value_missing_fields(self):
        """Test validation of return value missing fields."""
        return_value = {
            # Missing type and description
        }

        loader = ToolsLoader()

        with pytest.raises(
            ValueError, match="Return at index 0 missing required field: type"
        ):
            loader._validate_parameter(return_value, 0, "Return")


class TestToolsLoaderIntegration:
    """Integration tests for ToolsLoader."""

    def test_complete_workflow_invalid_tools(self):
        """Test complete workflow with invalid tools."""
        tools_json = json.dumps(
            [
                {
                    "name": "valid_tool",
                    "description": "A valid tool",
                    "parameters": [],
                    "return": {"type": "string", "description": "Result"},
                },
                {
                    "name": "invalid_tool",
                    "description": "An invalid tool",
                    # Missing parameters and return
                },
            ]
        )

        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status") as mock_print:
            result = loader.load_tools_from_json_string(tools_json)

        assert result["success"] is False
        assert result["tools"] == []

        # Verify error messages were printed
        error_calls = [call for call in mock_print.call_args_list if "has issues" in str(call)]
        assert len(error_calls) > 0


class TestToolsLoaderErrorHandling:
    """Test error handling in ToolsLoader."""

    def test_exception_during_validation(self):
        """Test handling of exceptions during validation."""
        loader = ToolsLoader(use_colors=False)

        # Mock _validate_tools_collection to raise an exception
        with patch.object(
            loader, "_validate_tools_collection", side_effect=Exception("Validation error")
        ):
            with patch.object(loader, "_print_status") as mock_print:
                result = loader.load_tools_from_json_string("[]")

        assert result["success"] is False
        assert result["tools"] == []

        # Verify error message was printed
        error_calls = [
            call
            for call in mock_print.call_args_list
            if "Error loading tools from JSON" in str(call)
        ]
        assert len(error_calls) > 0

    def test_json_decode_error_handling(self):
        """Test specific handling of JSON decode errors."""
        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status") as mock_print:
            result = loader.load_tools_from_json_string('{"invalid": json}')

        assert result["success"] is False
        assert result["tools"] == []

        # Verify JSON error message was printed
        json_error_calls = [
            call for call in mock_print.call_args_list if "Invalid JSON format" in str(call)
        ]
        assert len(json_error_calls) > 0

    def test_empty_string_input(self):
        """Test handling of empty string input."""
        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status") as mock_print:
            result = loader.load_tools_from_json_string("")

        assert result["success"] is False
        assert result["tools"] == []

    def test_none_input_handling(self):
        """Test handling of None input."""
        loader = ToolsLoader(use_colors=False)

        with patch.object(loader, "_print_status") as mock_print:
            result = loader.load_tools_from_json_string(None)

        assert result["success"] is False
        assert result["tools"] == []

        # Verify error message was printed
        error_calls = [
            call
            for call in mock_print.call_args_list
            if "Error loading tools from JSON" in str(call)
        ]
        assert len(error_calls) > 0
