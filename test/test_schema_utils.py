"""
Tests for data_schema/utils.py
"""

import json
from unittest.mock import mock_open, patch

import pytest

from elastic_gumby_universal_orch_agent_prototype.data_schema.utils import (
    get_workflow_schema,
    get_tools_schema
)


class TestGetWorkflowSchema:
    """Test get_workflow_schema function."""

    def test_get_workflow_schema_success(self):
        """Test successful loading of workflow schema."""
        # Mock schema data
        mock_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Workflow Orchestration Schema",
            "type": "object",
            "required": ["name", "description", "root"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "root": {"type": "object"}
            }
        }
        
        mock_file_content = json.dumps(mock_schema)
        
        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = get_workflow_schema()
        
        assert result == mock_schema
        assert result["title"] == "Workflow Orchestration Schema"
        assert "name" in result["properties"]
        assert "description" in result["properties"]
        assert "root" in result["properties"]

    def test_get_workflow_schema_json_decode_error(self):
        """Test handling of JSON decode error in workflow schema."""
        # Mock invalid JSON content
        invalid_json = '{"invalid": json content}'
        
        with patch("builtins.open", mock_open(read_data=invalid_json)):
            with pytest.raises(json.JSONDecodeError):
                get_workflow_schema()

    def test_get_workflow_schema_file_not_found(self):
        """Test handling of file not found error."""
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                get_workflow_schema()

class TestGetToolsSchema:
    """Test get_tools_schema function."""

    def test_get_tools_schema_success(self):
        """Test successful loading of tools schema."""
        # Mock schema data
        mock_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Workflow Orchestration Tool Input Schema",
            "type": "object",
            "required": ["available_tools"],
            "properties": {
                "available_tools": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/ToolDefinition"}
                }
            },
            "definitions": {
                "ToolDefinition": {
                    "type": "object",
                    "required": ["name", "description", "resource", "parameters", "return"]
                }
            }
        }
        
        mock_file_content = json.dumps(mock_schema)
        
        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            result = get_tools_schema()
        
        assert result == mock_schema
        assert result["title"] == "Workflow Orchestration Tool Input Schema"
        assert "available_tools" in result["properties"]
        assert "ToolDefinition" in result["definitions"]

    def test_get_tools_schema_json_decode_error(self):
        """Test handling of JSON decode error in tools schema."""
        # Mock invalid JSON content
        invalid_json = '{"malformed": json, content}'
        
        with patch("builtins.open", mock_open(read_data=invalid_json)):
            with pytest.raises(json.JSONDecodeError):
                get_tools_schema()

    def test_get_tools_schema_file_not_found(self):
        """Test handling of file not found error."""
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                get_tools_schema()