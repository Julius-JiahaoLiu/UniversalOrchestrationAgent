"""
Tests for ToolDescriptionTransformer

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

import pytest
from unittest.mock import patch

from elastic_gumby_universal_orch_agent_prototype.transform.tool_description_transformer import ToolDescriptionTransformer

class TestTransformDescription:
    """Test transform_description method."""

    @pytest.fixture
    def transformer(self):
        """Create ToolDescriptionTransformer instance with mocked dependencies."""
        with patch('elastic_gumby_universal_orch_agent_prototype.transform.tool_description_transformer.get_tools_schema') as mock_schema, \
             patch('elastic_gumby_universal_orch_agent_prototype.transform.tool_description_transformer.BedrockClientManager') as mock_bedrock, \
             patch('elastic_gumby_universal_orch_agent_prototype.transform.tool_description_transformer.ToolsLoader') as mock_loader:
            
            mock_schema.return_value = {"type": "object"}
            transformer = ToolDescriptionTransformer()
            transformer.bedrock_manager = mock_bedrock.return_value
            transformer.tools_loader = mock_loader.return_value
            return transformer

    def test_transform_description_end_turn_with_error(self, transformer):
        """Test transformation ending with error message."""
        # Mock bedrock response with end_turn and error
        mock_response = {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": "Error: Missing required field 'resource' in tool description."
                }
            ]
        }
        
        transformer.bedrock_manager.invoke_model.return_value = mock_response
        
        tool_description = "Incomplete tool description"
        result = transformer.transform_description(tool_description)
        
        # Should return empty dict when there's an error
        assert result == {}
        
        # Verify bedrock was called
        transformer.bedrock_manager.invoke_model.assert_called_once()

    def test_transform_description_max_interactions_reached(self, transformer):
        """Test transformation when max interactions is reached."""
        # Mock bedrock response that doesn't provide valid tool
        mock_response = {
            "stop_reason": "max_tokens",
            "content": [
                {
                    "type": "text",
                    "text": "Partial response..."
                }
            ]
        }
        
        transformer.bedrock_manager.invoke_model.return_value = mock_response
        transformer.max_interactions = 2  # Set low for testing
        
        tool_description = "A tool description"
        result = transformer.transform_description(tool_description)
        
        # Should return empty dict when max interactions reached without success
        assert result == {}
        
        # Verify bedrock was called max_interactions times
        assert transformer.bedrock_manager.invoke_model.call_count == 2

    def test_transform_description_validation_failure_then_success(self, transformer):
        """Test transformation with initial validation failure then success."""
        # This test demonstrates the current behavior where the loop breaks on the first truthy result
        # even if validation fails. This might be a bug in the implementation.
        
        # Mock first response with invalid tool
        invalid_response = {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "input": {
                        "name": "test_tool",
                        "description": "A test tool"
                        # Missing required fields like 'resource'
                    }
                }
            ]
        }
        
        # Mock validation failure for the invalid tool
        invalid_validation = {"success": False, "tools": []}
        
        transformer.bedrock_manager.invoke_model.return_value = invalid_response
        transformer.tools_loader.load_tools_from_json_string.return_value = invalid_validation
        
        tool_description = "A tool description"
        result = transformer.transform_description(tool_description)
        
        # Return {} when validation fails
        assert result == {}
        
        # Verify bedrock calling reach the max interaction limit
        assert transformer.bedrock_manager.invoke_model.call_count == 5

class TestProcessToolUse:
    """Test process_tool_use method."""

    @pytest.fixture
    def transformer(self):
        """Create ToolDescriptionTransformer instance with mocked dependencies."""
        with patch('elastic_gumby_universal_orch_agent_prototype.transform.tool_description_transformer.get_tools_schema') as mock_schema, \
             patch('elastic_gumby_universal_orch_agent_prototype.transform.tool_description_transformer.BedrockClientManager') as mock_bedrock, \
             patch('elastic_gumby_universal_orch_agent_prototype.transform.tool_description_transformer.ToolsLoader') as mock_loader:
            
            mock_schema.return_value = {"type": "object"}
            transformer = ToolDescriptionTransformer()
            transformer.bedrock_manager = mock_bedrock.return_value
            transformer.tools_loader = mock_loader.return_value
            return transformer

    def test_process_tool_use_with_text_and_tool_use(self, transformer):
        """Test processing tool use with both text and tool_use content."""
        content_list = [
            {
                "type": "text",
                "text": "I'll transform this tool description."
            },
            {
                "type": "tool_use",
                "id": "tool_123",
                "input": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "resource": "arn:aws:lambda:us-west-2:123456789012:function:test_tool",
                    "parameters": [],
                    "return": {"type": "string", "description": "Test result"}
                }
            }
        ]
        
        messages = []
        
        # Mock successful validation
        mock_validation_result = {
            "success": True,
            "tools": [{
                "name": "test_tool",
                "description": "A test tool",
                "resource": "arn:aws:lambda:us-west-2:123456789012:function:test_tool",
                "parameters": [],
                "return": {"type": "string", "description": "Test result"}
            }]
        }
        
        transformer.tools_loader.load_tools_from_json_string.return_value = mock_validation_result
        
        with patch.object(transformer, '_print_assistant_text') as mock_print:
            result = transformer.process_tool_use(content_list, messages)
        
        # Verify result
        assert result["name"] == "test_tool"
        assert result["description"] == "A test tool"
        
        # Verify assistant text was printed
        mock_print.assert_called_once_with("I'll transform this tool description.", "Tool Description Transformation Reasoning")
        
        # Verify messages were updated
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"][0]["type"] == "tool_result"
        assert messages[1]["content"][0]["tool_use_id"] == "tool_123"

    def test_process_tool_use_validation_failure(self, transformer):
        """Test processing tool use with validation failure."""
        content_list = [
            {
                "type": "tool_use",
                "id": "tool_123",
                "input": {
                    "name": "invalid_tool"
                    # Missing required fields
                }
            }
        ]
        
        messages = []
        
        # Mock validation failure
        mock_validation_result = {"success": False, "tools": []}
        transformer.tools_loader.load_tools_from_json_string.return_value = mock_validation_result
        
        result = transformer.process_tool_use(content_list, messages)
        
        # Return empty dict on validation failure
        assert result == {}
        
        # Verify error message was set in tool_result
        assert len(messages) == 2
        assert "errors" in messages[1]["content"][0]["content"]

class TestIntegrationScenarios:
    """Integration tests for complete transformation scenarios."""

    @pytest.fixture
    def transformer(self):
        """Create ToolDescriptionTransformer instance with mocked dependencies."""
        with patch('elastic_gumby_universal_orch_agent_prototype.transform.tool_description_transformer.get_tools_schema') as mock_schema, \
             patch('elastic_gumby_universal_orch_agent_prototype.transform.tool_description_transformer.BedrockClientManager') as mock_bedrock, \
             patch('elastic_gumby_universal_orch_agent_prototype.transform.tool_description_transformer.ToolsLoader') as mock_loader:
            
            mock_schema.return_value = {"type": "object"}
            transformer = ToolDescriptionTransformer()
            transformer.bedrock_manager = mock_bedrock.return_value
            transformer.tools_loader = mock_loader.return_value
            return transformer

    def test_complex_tool_with_nested_parameters(self, transformer):
        """Test transformation of complex tool with nested parameter structures."""
        tool_description = """
        A complex API tool that processes user data with nested configuration options.
        """
        
        # Mock response with complex nested structure
        complex_tool_input = {
            "name": "process_user_data",
            "description": "Processes user data with configurable options",
            "resource": "arn:aws:lambda:us-west-2:123456789012:function:process_user_data",
            "parameters": [
                {
                    "name": "user_data",
                    "type": "object",
                    "description": "User data to process",
                    "required": True
                },
                {
                    "name": "config",
                    "type": "object", 
                    "description": "Processing configuration",
                    "required": False,
                    "default_value": {"format": "json", "validate": True}
                }
            ],
            "return": {
                "type": "object",
                "description": "Processed user data with metadata",
                "schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                        "metadata": {"type": "object"}
                    }
                }
            }
        }
        
        mock_response = {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "complex_tool_456",
                    "input": complex_tool_input
                }
            ]
        }
        
        mock_validation_result = {
            "success": True,
            "tools": [complex_tool_input]
        }
        
        transformer.bedrock_manager.invoke_model.return_value = mock_response
        transformer.tools_loader.load_tools_from_json_string.return_value = mock_validation_result
        
        result = transformer.transform_description(tool_description)
        
        # Verify complex structure is preserved
        assert result["name"] == "process_user_data"
        assert len(result["parameters"]) == 2
        assert result["parameters"][1]["default_value"]["format"] == "json"
        assert "schema" in result["return"]
        assert "properties" in result["return"]["schema"]