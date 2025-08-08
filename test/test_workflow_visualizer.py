"""
Tests for WorkflowVisualizer

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from elastic_gumby_universal_orch_agent_prototype.visualizer.workflow_visualizer import WorkflowVisualizer


class TestVisualizeWorkflowBasic:
    """Test visualize_workflow method with basic workflow structures."""

    def test_visualize_workflow_minimal_fields(self):
        """Test visualization with minimal workflow fields."""
        workflow = {
            "root": {
                "type": "tool_call",
                "toolName": "simple_tool"
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        # Should handle missing name/description gracefully
        assert "WORKFLOW: Unnamed Workflow" in result
        assert "Description: No description" in result
        assert "TOOL_CALL" in result
        assert "simple_tool" in result

    def test_visualize_workflow_no_colors(self):
        """Test visualization without colors."""
        workflow = {
            "name": "Test Workflow",
            "description": "Test description",
            "root": {
                "type": "tool_call",
                "toolName": "test_tool"
            }
        }
        
        visualizer = WorkflowVisualizer(use_colors=False)
        result = visualizer.visualize_workflow(workflow)
        
        # Should not contain ANSI color codes
        assert "\033[" not in result
        assert "TOOL_CALL" in result
        assert "test_tool" in result

    def test_visualize_workflow_no_icons(self):
        """Test visualization without icons."""
        workflow = {
            "name": "Test Workflow",
            "description": "Test description",
            "root": {
                "type": "tool_call",
                "toolName": "test_tool"
            }
        }
        
        visualizer = WorkflowVisualizer(use_icons=False)
        result = visualizer.visualize_workflow(workflow)
        
        # Should not contain Unicode icons when use_icons=False
        assert "🔧" not in result
        assert "TOOL_CALL" in result
        assert "test_tool" in result


class TestVisualizeWorkflowToolCall:
    """Test visualize_workflow method with tool_call nodes."""

    def test_visualize_tool_call_no_parameters(self):
        """Test visualization of tool_call without parameters."""
        workflow = {
            "name": "No Params Tool",
            "root": {
                "type": "tool_call",
                "toolName": "no_param_tool"
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        assert "no_param_tool" in result
        # When no parameters are provided, no parameter info is shown
        assert "TOOL_CALL" in result

    def test_visualize_tool_call_complex_parameters(self):
        """Test visualization of tool_call with complex parameters."""
        workflow = {
            "name": "Complex Params Tool",
            "root": {
                "type": "tool_call",
                "toolName": "complex_tool",
                "parameters": {
                    "var_param1": "{% $var1 %}",
                    "var_param2": "{% $item1 %}",
                    "simple_var": "{% $simple %}",
                    "static_string": "just_text"
                }
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        assert "complex_tool" in result
        assert "$var1" in result
        assert "$item1" in result
        assert "$simple" in result
        assert "static param" in result


class TestVisualizeWorkflowWaitForEvent:
    """Test visualize_workflow method with wait_for_event nodes."""

    def test_visualize_wait_for_event_with_timeout_handler(self):
        """Test visualization of wait_for_event with timeout handler."""
        workflow = {
            "name": "Wait with Timeout",
            "root": {
                "type": "wait_for_event",
                "eventSource": "lambda",
                "eventType": "function_complete",
                "timeout": 60,
                "onTimeout": {
                    "type": "tool_call",
                    "toolName": "handle_timeout"
                }
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        assert "WAIT_EVENT" in result
        assert "lambda.function_complete" in result
        assert "timeout: 60s" in result
        assert "ON TIMEOUT:" in result
        assert "handle_timeout" in result

    def test_visualize_wait_for_event_minimal(self):
        """Test visualization of minimal wait_for_event node."""
        workflow = {
            "name": "Minimal Wait",
            "root": {
                "type": "wait_for_event",
                "eventSource": "ec2",
                "eventType": "instance_ready"
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        assert "WAIT_EVENT" in result
        assert "ec2.instance_ready" in result
        # Should not contain timeout or output variable info
        assert "timeout:" not in result
        assert "→" not in result


class TestVisualizeWorkflowUserInput:
    """Test visualize_workflow method with user_input nodes."""

    def test_visualize_user_input_with_options(self):
        """Test visualization of user_input with options."""
        workflow = {
            "name": "Choice Input",
            "root": {
                "type": "user_input",
                "prompt": "Select an option",
                "inputType": "choice",
                "options": ["Option A", "Option B", "Option C"],
                "outputVariable": "choice"
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        assert "USER_INPUT" in result
        assert "Select an option" in result
        assert "Type: choice" in result
        assert "Options: ['Option A', 'Option B', 'Option C']" in result
        assert "→ choice" in result

    def test_visualize_user_input_with_variables(self):
        """Test visualization of user_input with variable in prompt."""
        workflow = {
            "name": "Dynamic Prompt",
            "root": {
                "type": "user_input",
                "prompt": "Hello {% $user_name %}, please enter {% $field_name %}",
                "inputType": "text"
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        assert "USER_INPUT" in result
        assert "$user_name" in result
        assert "$field_name" in result


class TestVisualizeWorkflowBranch:
    """Test visualize_workflow method with branch nodes."""

    def test_visualize_branch_logical_condition(self):
        """Test visualization of branch with logical condition."""
        workflow = {
            "name": "Logical Branch",
            "root": {
                "type": "branch",
                "condition": {
                    "type": "logical",
                    "operator": "and",
                    "conditions": [
                        {
                            "type": "comparison",
                            "left": "{% $count %}",
                            "operator": ">",
                            "right": "0"
                        },
                        {
                            "type": "comparison",
                            "left": "{% $enabled %}",
                            "operator": "==",
                            "right": "true"
                        }
                    ]
                },
                "ifTrue": {
                    "type": "tool_call",
                    "toolName": "process_items"
                },
                "ifFalse": {
                    "type": "tool_call",
                    "toolName": "skip_processing"
                }
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        assert "BRANCH" in result
        assert "$count" in result
        assert "> 0" in result
        assert "$enabled" in result
        assert "== true" in result
        assert "AND" in result
        assert "process_items" in result
        assert "skip_processing" in result

    def test_visualize_branch_not_condition(self):
        """Test visualization of branch with NOT condition."""
        workflow = {
            "name": "NOT Branch",
            "root": {
                "type": "branch",
                "condition": {
                    "type": "logical",
                    "operator": "not",
                    "conditions": [
                        {
                            "type": "comparison",
                            "left": "{% $ready %}",
                            "operator": "==",
                            "right": "false"
                        }
                    ]
                },
                "ifTrue": {
                    "type": "tool_call",
                    "toolName": "proceed"
                },
                "ifFalse": {
                    "type": "tool_call",
                    "toolName": "wait"
                }
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        assert "BRANCH" in result
        assert "NOT" in result
        assert "$ready" in result
        assert "== false" in result


class TestVisualizeWorkflowLoop:
    """Test visualize_workflow method with loop nodes."""

    def test_visualize_loop_basic(self):
        """Test visualization of basic loop node."""
        workflow = {
            "name": "Loop Workflow",
            "root": {
                "type": "loop",
                "condition": {
                    "type": "comparison",
                    "left": "{% $counter %}",
                    "operator": "<",
                    "right": "10"
                },
                "iterationVariable": "i",
                "body": {
                    "type": "tool_call",
                    "toolName": "process_item"
                }
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        assert "LOOP" in result
        assert "$counter" in result
        assert "< 10" in result
        assert "iter → i" in result
        assert "LOOP BODY:" in result
        assert "process_item" in result

class TestVisualizeWorkflowSequence:
    """Test visualize_workflow method with sequence nodes."""

    def test_visualize_sequence_basic(self):
        """Test visualization of basic sequence node."""
        workflow = {
            "name": "Sequence Workflow",
            "root": {
                "type": "sequence",
                "description": "Execute steps in order",
                "steps": [
                    {
                        "type": "tool_call",
                        "toolName": "step_one"
                    },
                    {
                        "type": "tool_call",
                        "toolName": "step_two"
                    },
                    {
                        "type": "tool_call",
                        "toolName": "step_three"
                    }
                ]
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        assert "SEQUENCE" in result
        assert "Execute steps in order" in result
        assert "step_one" in result
        assert "step_two" in result
        assert "step_three" in result
        # Check for proper tree structure
        assert "├──" in result
        assert "└──" in result

class TestVisualizeWorkflowParallel:
    """Test visualize_workflow method with parallel nodes."""

    def test_visualize_parallel_basic(self):
        """Test visualization of basic parallel node."""
        workflow = {
            "name": "Parallel Workflow",
            "root": {
                "type": "parallel",
                "description": "Execute branches concurrently",
                "maxConcurrency": 3,
                "aggregateVariable": "results",
                "branches": [
                    {
                        "type": "tool_call",
                        "toolName": "branch_one"
                    },
                    {
                        "type": "tool_call",
                        "toolName": "branch_two"
                    }
                ]
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        assert "PARALLEL" in result
        assert "Execute branches concurrently" in result
        assert "max: 3" in result
        assert "→ results" in result
        assert "branch_one" in result
        assert "branch_two" in result

class TestVisualizeWorkflowComplex:
    """Test visualize_workflow method with complex nested structures."""

    def test_visualize_nested_workflow(self):
        """Test visualization of complex nested workflow."""
        workflow = {
            "name": "Complex Nested Workflow",
            "description": "A workflow with multiple nested structures",
            "root": {
                "type": "sequence",
                "steps": [
                    {
                        "type": "tool_call",
                        "toolName": "initialize",
                        "outputVariable": "init_result"
                    },
                    {
                        "type": "branch",
                        "condition": {
                            "type": "comparison",
                            "left": "{% $init_result %}",
                            "operator": "==",
                            "right": "success"
                        },
                        "ifTrue": {
                            "type": "parallel",
                            "branches": [
                                {
                                    "type": "tool_call",
                                    "toolName": "process_a"
                                },
                                {
                                    "type": "tool_call",
                                    "toolName": "process_b"
                                }
                            ]
                        },
                        "ifFalse": {
                            "type": "tool_call",
                            "toolName": "handle_error"
                        }
                    },
                    {
                        "type": "tool_call",
                        "toolName": "finalize"
                    }
                ]
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        # Verify all components are present
        assert "Complex Nested Workflow" in result
        assert "SEQUENCE" in result
        assert "initialize" in result
        assert "BRANCH" in result
        assert "$init_result" in result
        assert "== success" in result
        assert "PARALLEL" in result
        assert "process_a" in result
        assert "process_b" in result
        assert "handle_error" in result
        assert "finalize" in result

    def test_visualize_deeply_nested_workflow(self):
        """Test visualization of deeply nested workflow structures."""
        workflow = {
            "name": "Deep Nesting",
            "root": {
                "type": "sequence",
                "steps": [
                    {
                        "type": "loop",
                        "condition": {
                            "type": "comparison",
                            "left": "{% $i %}",
                            "operator": "<",
                            "right": "5"
                        },
                        "body": {
                            "type": "branch",
                            "condition": {
                                "type": "comparison",
                                "left": "{% $i %}",
                                "operator": "%",
                                "right": "2"
                            },
                            "ifTrue": {
                                "type": "tool_call",
                                "toolName": "process_even"
                            },
                            "ifFalse": {
                                "type": "tool_call",
                                "toolName": "process_odd"
                            }
                        }
                    }
                ]
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        assert "SEQUENCE" in result
        assert "LOOP" in result
        assert "BRANCH" in result
        assert "process_even" in result
        assert "process_odd" in result


class TestVisualizeWorkflowUnknownNode:
    """Test visualize_workflow method with unknown node types."""

    def test_visualize_unknown_node_type(self):
        """Test visualization with unknown node type."""
        workflow = {
            "name": "Unknown Node Workflow",
            "root": {
                "type": "unknown_type",
                "someProperty": "someValue"
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        # Should handle unknown node types gracefully
        assert "unknown_type" in result
        assert "Unknown Node Workflow" in result


class TestSaveWorkflowVisualization:
    """Test save_workflow_visualization method."""

    def test_save_workflow_visualization_basic(self):
        """Test saving workflow visualization to file."""
        workflow = {
            "name": "Test Save Workflow",
            "description": "Testing file save functionality",
            "root": {
                "type": "tool_call",
                "toolName": "test_tool"
            }
        }
        
        visualizer = WorkflowVisualizer(use_colors=True)
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as temp_file:
            temp_path = temp_file.name
        
        try:
            with patch("builtins.print"):
                visualizer.save_workflow_visualization(workflow, temp_path)
            
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Should not contain ANSI escape codes
            assert "\033[" not in content
            assert "Test Save Workflow" in content
            assert "test_tool" in content
            assert "🔄Workflow Visualization" in content  # Icon is included in the header
            assert "Generated:" in content
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_save_workflow_visualization_file_error(self, mock_open):
        """Test save_workflow_visualization with file write error."""
        workflow = {
            "name": "Test Workflow",
            "root": {"type": "tool_call", "toolName": "test"}
        }
        visualizer = WorkflowVisualizer()
        
        # Should raise the IOError
        with pytest.raises(IOError, match="Permission denied"):
            visualizer.save_workflow_visualization(workflow, "/invalid/path/file.md")


class TestWorkflowVisualizerEdgeCases:
    """Test edge cases and error conditions."""

    def test_visualize_workflow_missing_root(self):
        """Test visualization with missing root."""
        workflow = {
            "name": "Missing Root Workflow"
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        # Should handle missing root gracefully
        assert "Missing Root Workflow" in result

    def test_visualize_workflow_malformed_condition(self):
        """Test visualization with malformed condition."""
        workflow = {
            "name": "Malformed Condition",
            "root": {
                "type": "branch",
                "condition": {
                    "type": "unknown_condition_type"
                },
                "ifTrue": {
                    "type": "tool_call",
                    "toolName": "true_branch"
                },
                "ifFalse": {
                    "type": "tool_call",
                    "toolName": "false_branch"
                }
            }
        }
        
        visualizer = WorkflowVisualizer()
        result = visualizer.visualize_workflow(workflow)
        
        # Should handle unknown condition types
        assert "Unknown condition" in result
        assert "true_branch" in result
        assert "false_branch" in result

    def test_format_parameters_edge_cases(self):
        """Test _format_parameters method with edge cases."""
        visualizer = WorkflowVisualizer()
        
        # Test with empty parameters
        result = visualizer._format_parameters({})
        assert result == ""  # Empty dict returns empty string
        
        # Test with None values
        result = visualizer._format_parameters({"key": None})
        assert "static param" in result
        
        # Test with mixed types
        result = visualizer._format_parameters({
            "string_var": "{% $test %}",
            "number": 42,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        })
        assert "$test" in result
        assert "static param" in result
