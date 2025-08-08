"""
Tests for WorkflowProcessor

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

from unittest.mock import patch

from elastic_gumby_universal_orch_agent_prototype.planner.workflow_processor import (
    WorkflowProcessor,
)


class TestExtractFinalMetadata:
    """Test extract_final_metadata method."""

    def test_extract_final_metadata_valid_json_format(self):
        """Test extracting metadata from valid JSON-like format."""
        processor = WorkflowProcessor()

        text_response = '{"name": "Test Workflow", "description": "A test workflow for validation"}'

        result = processor.extract_final_metadata(text_response)

        assert result == {"name": "Test Workflow", "description": "A test workflow for validation"}

    def test_extract_final_metadata_with_whitespace(self):
        """Test extracting metadata with extra whitespace."""
        processor = WorkflowProcessor()

        text_response = """
        {
            "name"  :  "Spaced Workflow"  ,
            "description"  :  "Workflow with spaces"
        }
        """

        result = processor.extract_final_metadata(text_response)

        assert result == {"name": "Spaced Workflow", "description": "Workflow with spaces"}

    def test_extract_final_metadata_partial_match(self):
        """Test extracting metadata when only one field is present."""
        processor = WorkflowProcessor()

        text_response = '{"name": "Partial Workflow"}'

        result = processor.extract_final_metadata(text_response)

        # Should return empty dict when both fields are not present
        assert result == {}

    def test_extract_final_metadata_no_match(self):
        """Test extracting metadata when no valid format is found."""
        processor = WorkflowProcessor()

        text_response = "This is just plain text without the expected format"

        result = processor.extract_final_metadata(text_response)

        assert result == {}

    def test_extract_final_metadata_malformed_json(self):
        """Test extracting metadata from malformed JSON."""
        processor = WorkflowProcessor()

        text_response = '{"name": "Malformed", "description": "Missing quote}'

        result = processor.extract_final_metadata(text_response)

        assert result == {}

    def test_extract_final_metadata_empty_string(self):
        """Test extracting metadata from empty string."""
        processor = WorkflowProcessor()

        result = processor.extract_final_metadata("")

        assert result == {}

    def test_extract_final_metadata_none_input(self):
        """Test extracting metadata from None input."""
        processor = WorkflowProcessor()

        result = processor.extract_final_metadata(None)

        # Should return empty dict and handle the error gracefully
        assert result == {}

    @patch("builtins.print")
    def test_extract_final_metadata_warning_message(self, mock_print):
        """Test that warning message is printed when metadata extraction fails."""
        processor = WorkflowProcessor()

        text_response = "No valid metadata here"

        processor.extract_final_metadata(text_response)

        # Verify warning message was printed
        mock_print.assert_called()
        printed_args = [call.args[0] for call in mock_print.call_args_list]
        warning_messages = [msg for msg in printed_args if "Warning: Could not extract" in msg]
        assert len(warning_messages) == 1


class TestCombineWorkflowSections:
    """Test combine_workflow_sections method."""

    def test_combine_workflow_sections_empty_list(self):
        """Test combining empty workflow sections list."""
        processor = WorkflowProcessor()

        result = processor.combine_workflow_sections([])

        assert result == {"error": "No workflow sections to combine"}

    def test_combine_workflow_sections_multiple_sections(self):
        """Test combining multiple workflow sections."""
        processor = WorkflowProcessor()

        workflow_sections = [
            {
                "workflow_plan": {
                    "name": "First Workflow",
                    "description": "First workflow description",
                    "root": {"type": "tool_call", "name": "tool1", "description": "First tool"},
                },
                "section_number": 1,
            },
            {
                "workflow_plan": {
                    "name": "Second Workflow",
                    "description": "Second workflow description",
                    "root": {"type": "tool_call", "name": "tool2", "description": "Second tool"},
                },
                "section_number": 2,
            },
        ]

        result = processor.combine_workflow_sections(workflow_sections)

        # Should create a sequence with both tools
        assert result["name"] == "First Workflow"
        assert result["description"] == "First workflow description"
        assert result["root"]["type"] == "sequence"
        assert len(result["root"]["steps"]) == 2
        assert result["root"]["steps"][0]["name"] == "tool1"
        assert result["root"]["steps"][1]["name"] == "tool2"

    def test_combine_workflow_sections_missing_root(self):
        """Test combining sections with missing root element."""
        processor = WorkflowProcessor()

        workflow_sections = [
            {
                "workflow_plan": {
                    "name": "Valid Workflow",
                    "description": "Valid workflow",
                    "root": {
                        "type": "tool_call",
                        "name": "valid_tool",
                        "description": "Valid tool",
                    },
                }
            },
            {
                "workflow_plan": {
                    "name": "Invalid Workflow",
                    "description": "Missing root",
                    # Missing root element
                },
                "section_number": 2,
            },
        ]

        with patch("builtins.print") as mock_print:
            result = processor.combine_workflow_sections(workflow_sections)

            # Should print warning for missing root
            printed_args = [call.args[0] for call in mock_print.call_args_list]
            warning_messages = [msg for msg in printed_args if "missing 'root' element" in msg]
            assert len(warning_messages) == 1

        # Should still process the valid section
        assert result["root"]["name"] == "valid_tool"

    def test_combine_workflow_sections_default_metadata(self):
        """Test combining sections with missing name/description."""
        processor = WorkflowProcessor()

        workflow_sections = [
            {
                "workflow_plan": {
                    # Missing name and description
                    "root": {"type": "tool_call", "name": "test_tool", "description": "Test tool"}
                }
            }
        ]

        result = processor.combine_workflow_sections(workflow_sections)

        # Should use default values
        assert result["name"] == "Combined Workflow"
        assert result["description"] == "Multi-section workflow execution"

    @patch("builtins.print")
    def test_combine_workflow_sections_print_messages(self, mock_print):
        """Test that appropriate messages are printed during combination."""
        processor = WorkflowProcessor()

        workflow_sections = [
            {
                "workflow_plan": {
                    "name": "Test Workflow",
                    "description": "Test description",
                    "root": {"type": "tool_call", "name": "test_tool", "description": "Test tool"},
                }
            }
        ]

        processor.combine_workflow_sections(workflow_sections)

        # Verify messages were printed
        mock_print.assert_called()
        printed_args = [call.args[0] for call in mock_print.call_args_list]

        # Should have combining message
        combining_messages = [
            msg for msg in printed_args if "Combining" in msg and "workflow sections" in msg
        ]
        assert len(combining_messages) == 1

        # Should have success message
        success_messages = [msg for msg in printed_args if "✓ Successfully combined" in msg]
        assert len(success_messages) == 1


class TestFlattenWorkflowSection:
    """Test flatten_workflow_section method and helper methods."""

    def test_flatten_workflow_section_non_dict_input(self):
        """Test flattening with non-dictionary input."""
        processor = WorkflowProcessor()

        result = processor.flatten_workflow_section("not a dict", 1)

        assert result == []

    def test_flatten_workflow_section_tool_call(self):
        """Test flattening a simple tool call node."""
        processor = WorkflowProcessor()

        workflow_node = {"type": "tool_call", "name": "test_tool", "description": "Test tool call"}

        result = processor.flatten_workflow_section(workflow_node, 1)

        assert result == [workflow_node]

    def test_flatten_workflow_section_sequence_empty(self):
        """Test flattening an empty sequence."""
        processor = WorkflowProcessor()

        workflow_node = {
            "type": "sequence",
            "name": "empty_sequence",
            "description": "Empty sequence",
            "steps": [],
        }

        result = processor.flatten_workflow_section(workflow_node, 1)

        assert result == []

    def test_flatten_workflow_section_nested_sequence(self):
        """Test flattening nested sequences."""
        processor = WorkflowProcessor()

        workflow_node = {
            "type": "sequence",
            "name": "outer_sequence",
            "description": "Outer sequence",
            "steps": [
                {"type": "tool_call", "name": "first_tool", "description": "First tool"},
                {
                    "type": "sequence",
                    "name": "inner_sequence",
                    "description": "Inner sequence",
                    "steps": [
                        {
                            "type": "tool_call",
                            "name": "nested_tool1",
                            "description": "First nested tool",
                        },
                        {
                            "type": "tool_call",
                            "name": "nested_tool2",
                            "description": "Second nested tool",
                        },
                    ],
                },
            ],
        }

        result = processor.flatten_workflow_section(workflow_node, 1)

        # Should flatten all steps
        assert len(result) == 3
        assert result[0]["name"] == "first_tool"
        assert result[1]["name"] == "nested_tool1"
        assert result[2]["name"] == "nested_tool2"

    def test_flatten_workflow_section_parallel_empty(self):
        """Test flattening an empty parallel node."""
        processor = WorkflowProcessor()

        workflow_node = {"type": "parallel", "description": "Empty parallel", "branches": []}

        result = processor.flatten_workflow_section(workflow_node, 1)

        assert result == []

    def test_flatten_workflow_section_parallel_multi_step_branches(self):
        """Test flattening parallel node with multi-step branches."""
        processor = WorkflowProcessor()

        workflow_node = {
            "type": "parallel",
            "description": "Complex parallel execution",
            "branches": [
                {
                    "type": "sequence",
                    "name": "branch1_sequence",
                    "description": "First branch sequence",
                    "steps": [
                        {
                            "type": "tool_call",
                            "name": "branch1_step1",
                            "description": "Branch 1 Step 1",
                        },
                        {
                            "type": "tool_call",
                            "name": "branch1_step2",
                            "description": "Branch 1 Step 2",
                        },
                    ],
                },
                {
                    "type": "tool_call",
                    "name": "branch2_single",
                    "description": "Branch 2 single step",
                },
            ],
        }

        result = processor.flatten_workflow_section(workflow_node, 2)

        assert len(result) == 1
        assert result[0]["type"] == "parallel"
        assert len(result[0]["branches"]) == 2

        # First branch should be wrapped in sequence
        first_branch = result[0]["branches"][0]
        assert first_branch["type"] == "sequence"
        assert first_branch["name"] == "Branch from section 2"
        assert len(first_branch["steps"]) == 2
        assert first_branch["steps"][0]["name"] == "branch1_step1"
        assert first_branch["steps"][1]["name"] == "branch1_step2"

        # Second branch should remain as single step
        second_branch = result[0]["branches"][1]
        assert second_branch["name"] == "branch2_single"
