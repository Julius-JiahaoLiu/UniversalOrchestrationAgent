"""
Tests for Phase2PlanningReflecting

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

import json
from unittest.mock import Mock, patch

import pytest

from elastic_gumby_universal_orch_agent_prototype.phases.phase2_planning_reflecting import Phase2PlanningReflecting

class TestGenerateWorkflowPlan:
    """Test generate_workflow_plan method."""

    @pytest.fixture
    def mock_session_data(self):
        """Create mock session data."""
        return {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": [],
                    "returns": {"name": "result", "type": "string", "description": "Test result"}
                }
            ],
            "workflow_plan": {},
            "claude_messages": []
        }

    @pytest.fixture
    def mock_get_user_input(self):
        """Create mock get_user_input function."""
        return Mock()

    @pytest.fixture
    def mock_workflow_visualizer(self):
        """Create mock workflow visualizer."""
        return Mock()

    @pytest.fixture
    def phase2_instance(self, mock_session_data, mock_get_user_input, mock_workflow_visualizer):
        """Create Phase2PlanningReflecting instance with mocked dependencies."""
        return Phase2PlanningReflecting(
            session_data=mock_session_data,
            get_user_input_func=mock_get_user_input,
            workflow_visualizer=mock_workflow_visualizer
        )

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase2_planning_reflecting.generate_plan')
    def test_generate_workflow_plan_success(self, mock_generate_plan, phase2_instance):
        """Test successful workflow plan generation."""
        # Set up the planner configuration attributes
        phase2_instance.model_id = "test-model-id"
        phase2_instance.max_interactions = 15
        phase2_instance.max_tokens = 6000
        
        mock_workflow_plan = {
            "name": "test_workflow",
            "description": "A test workflow",
            "root": {
                "type": "tool_call",
                "toolName": "write_file",
                "parameters": {
                    "path": "temp/test.py",
                    "content": "print('Hello World')"
                },
                "outputVariable": "file_result"
            }
        }
        mock_claude_messages = {"role": "assistant", "content": "Generated plan"}
        mock_generate_plan.return_value = (mock_workflow_plan, mock_claude_messages)

        with patch('builtins.print'):
            result = phase2_instance.generate_workflow_plan("Create a simple workflow")

        assert result is True
        mock_generate_plan.assert_called_once_with(
            workflow_description="Create a simple workflow",
            available_tools=phase2_instance.session_data["tools"],
            model_id="test-model-id",
            max_interactions=15,
            max_tokens=6000
        )
        assert phase2_instance.session_data["workflow_plan"] == mock_workflow_plan
        assert len(phase2_instance.session_data["claude_messages"]) == 1

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase2_planning_reflecting.generate_plan')
    def test_generate_workflow_plan_with_tools_resource_removal(self, mock_generate_plan, phase2_instance):
        """Test workflow plan generation removes 'resource' key from tools."""
        # Set up the planner configuration attributes
        phase2_instance.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        phase2_instance.max_interactions = 20
        phase2_instance.max_tokens = 8000
        
        phase2_instance.session_data["tools"] = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "resource": "some_resource_info",
                "parameters": []
            }
        ]
        
        mock_workflow_plan = {"name": "test_workflow"}
        mock_claude_messages = {"role": "assistant", "content": "Generated plan"}
        mock_generate_plan.return_value = (mock_workflow_plan, mock_claude_messages)

        with patch('builtins.print'):
            result = phase2_instance.generate_workflow_plan("Create a workflow")

        assert result is True
        called_tools = mock_generate_plan.call_args[1]['available_tools']
        assert all('resource' not in tool for tool in called_tools)
        # Verify original tools still have 'resource' key (deep copy was used)
        assert 'resource' in phase2_instance.session_data["tools"][0]
        
        # Verify all parameters were passed
        mock_generate_plan.assert_called_once_with(
            workflow_description="Create a workflow",
            available_tools=called_tools,
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_interactions=20,
            max_tokens=8000
        )

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase2_planning_reflecting.generate_plan')
    def test_generate_workflow_plan_exception(self, mock_generate_plan, phase2_instance):
        """Test workflow plan generation with exception."""
        # Set up the planner configuration attributes
        phase2_instance.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        phase2_instance.max_interactions = 20
        phase2_instance.max_tokens = 8000
        
        mock_generate_plan.side_effect = Exception("Planning failed")

        with patch('builtins.print'):
            result = phase2_instance.generate_workflow_plan("Create a workflow")

        assert result is False
        mock_generate_plan.assert_called_once()

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase2_planning_reflecting.generate_plan')
    def test_generate_workflow_plan_empty_tools(self, mock_generate_plan, phase2_instance):
        """Test workflow plan generation with empty tools list."""
        # Set up the planner configuration attributes
        phase2_instance.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        phase2_instance.max_interactions = 20
        phase2_instance.max_tokens = 8000
        
        phase2_instance.session_data["tools"] = []
        mock_workflow_plan = {"name": "test_workflow"}
        mock_claude_messages = {"role": "assistant", "content": "Generated plan"}
        mock_generate_plan.return_value = (mock_workflow_plan, mock_claude_messages)

        with patch('builtins.print'):
            result = phase2_instance.generate_workflow_plan("Create a workflow")

        assert result is True
        mock_generate_plan.assert_called_once_with(
            workflow_description="Create a workflow",
            available_tools=[],
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_interactions=20,
            max_tokens=8000
        )
        assert phase2_instance.session_data["workflow_plan"] == mock_workflow_plan


class TestReflectWorkflowPlan:
    """Test reflect_workflow_plan method."""

    @pytest.fixture
    def mock_session_data(self):
        """Create mock session data with existing workflow plan."""
        return {
            "tools": [
                {
                    "name": "data_processor",
                    "description": "Processes data",
                    "parameters": [],
                    "returns": {"name": "result", "type": "object", "description": "Processed data"}
                }
            ],
            "workflow_plan": {
                "name": "existing_workflow",
                "description": "An existing workflow",
                "root": {
                    "type": "tool_call",
                    "toolName": "data_processor",
                    "parameters": {
                        "input_data": "raw_data"
                    },
                    "outputVariable": "processed_data"
                }
            },
            "claude_messages": []
        }

    @pytest.fixture
    def mock_get_user_input(self):
        """Create mock get_user_input function."""
        return Mock()

    @pytest.fixture
    def mock_workflow_visualizer(self):
        """Create mock workflow visualizer."""
        return Mock()

    @pytest.fixture
    def phase2_instance(self, mock_session_data, mock_get_user_input, mock_workflow_visualizer):
        """Create Phase2PlanningReflecting instance with mocked dependencies."""
        return Phase2PlanningReflecting(
            session_data=mock_session_data,
            get_user_input_func=mock_get_user_input,
            workflow_visualizer=mock_workflow_visualizer
        )

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase2_planning_reflecting.reflect_plan')
    def test_reflect_workflow_plan_success(self, mock_reflect_plan, phase2_instance):
        """Test successful workflow plan reflection."""
        # Set up the planner configuration attributes
        phase2_instance.model_id = "test-model-id"
        phase2_instance.max_interactions = 15
        phase2_instance.max_tokens = 6000
        
        # Store the original workflow plan before the method call
        original_workflow_plan = phase2_instance.session_data["workflow_plan"].copy()
        
        updated_plan = {
            "name": "updated_workflow",
            "description": "Updated workflow based on feedback",
            "root": {
                "type": "tool_call",
                "toolName": "data_processor",
                "parameters": {
                    "input_data": "raw_data",
                    "additional_param": "value"
                },
                "outputVariable": "processed_data"
            }
        }
        mock_claude_messages = {"role": "assistant", "content": "Reflected plan"}
        mock_reflect_plan.return_value = (updated_plan, mock_claude_messages)

        feedback = "Add a data processing step between start and end"

        with patch('builtins.print'):
            result = phase2_instance.reflect_workflow_plan(feedback)

        assert result is True
        called_tools = mock_reflect_plan.call_args[1]['available_tools']
        assert all('resource' not in tool for tool in called_tools)
        existing_plan_passed = mock_reflect_plan.call_args[1]['existing_workflow_plan']
        assert existing_plan_passed == original_workflow_plan
        mock_reflect_plan.assert_called_once_with(
            existing_workflow_plan=original_workflow_plan,
            user_feedback=feedback,
            available_tools=called_tools,
            model_id="test-model-id",
            max_interactions=15,
            max_tokens=6000
        )
        # Verify session data was updated with the new plan
        assert phase2_instance.session_data["workflow_plan"] == updated_plan
        assert len(phase2_instance.session_data["claude_messages"]) == 1

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase2_planning_reflecting.reflect_plan')
    def test_reflect_workflow_plan_with_tools_resource_removal(self, mock_reflect_plan, phase2_instance):
        """Test workflow plan reflection removes 'resource' key from tools."""
        # Set up the planner configuration attributes
        phase2_instance.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        phase2_instance.max_interactions = 20
        phase2_instance.max_tokens = 8000
        
        # Store the original workflow plan before the method call
        original_workflow_plan = phase2_instance.session_data["workflow_plan"].copy()
        
        # Add tools with 'resource' key
        phase2_instance.session_data["tools"] = [
            {
                "name": "data_processor",
                "description": "Processes data",
                "resource": "some_resource_info",
                "parameters": []
            }
        ]
        
        updated_plan = {"name": "updated_workflow"}
        mock_claude_messages = {"role": "assistant", "content": "Reflected plan"}
        mock_reflect_plan.return_value = (updated_plan, mock_claude_messages)

        feedback = "Make some changes"

        with patch('builtins.print'):
            result = phase2_instance.reflect_workflow_plan(feedback)

        assert result is True
        called_tools = mock_reflect_plan.call_args[1]['available_tools']
        assert all('resource' not in tool for tool in called_tools)
        assert 'resource' in phase2_instance.session_data["tools"][0]
        
        # Verify all parameters were passed with the original workflow plan
        mock_reflect_plan.assert_called_once_with(
            existing_workflow_plan=original_workflow_plan,
            user_feedback=feedback,
            available_tools=called_tools,
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_interactions=20,
            max_tokens=8000
        )

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase2_planning_reflecting.reflect_plan')
    def test_reflect_workflow_plan_exception(self, mock_reflect_plan, phase2_instance):
        """Test workflow plan reflection with exception."""
        # Set up the planner configuration attributes
        phase2_instance.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        phase2_instance.max_interactions = 20
        phase2_instance.max_tokens = 8000
        
        mock_reflect_plan.side_effect = Exception("Reflection failed")

        feedback = "Add error handling"

        with patch('builtins.print'):
            result = phase2_instance.reflect_workflow_plan(feedback)

        assert result is False
        mock_reflect_plan.assert_called_once()

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase2_planning_reflecting.reflect_plan')
    def test_reflect_workflow_plan_no_existing_plan(self, mock_reflect_plan, phase2_instance):
        """Test workflow plan reflection with no existing plan."""
        # Set up the planner configuration attributes
        phase2_instance.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        phase2_instance.max_interactions = 20
        phase2_instance.max_tokens = 8000
        
        phase2_instance.session_data["workflow_plan"] = {}

        updated_plan = {"name": "new_workflow"}
        mock_claude_messages = {"role": "assistant", "content": "Created new plan"}
        mock_reflect_plan.return_value = (updated_plan, mock_claude_messages)

        feedback = "Create a new workflow"

        with patch('builtins.print'):
            result = phase2_instance.reflect_workflow_plan(feedback)

        assert result is True
        called_tools = mock_reflect_plan.call_args[1]['available_tools']
        assert all('resource' not in tool for tool in called_tools)
        mock_reflect_plan.assert_called_once_with(
            existing_workflow_plan={},
            user_feedback=feedback,
            available_tools=called_tools,
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_interactions=20,
            max_tokens=8000
        )
        assert phase2_instance.session_data["workflow_plan"] == updated_plan


class TestPhase2Integration:
    """Integration tests for Phase2PlanningReflecting."""

    @pytest.fixture
    def mock_session_data(self):
        """Create mock session data."""
        return {
            "tools": [{"name": "test_tool", "description": "A test tool"}],
            "workflow_plan": {},
            "claude_messages": []
        }

    @pytest.fixture
    def mock_get_user_input(self):
        """Create mock get_user_input function."""
        return Mock()

    @pytest.fixture
    def mock_workflow_visualizer(self):
        """Create mock workflow visualizer."""
        mock_visualizer = Mock()
        mock_visualizer.visualize_workflow.return_value = "Mocked workflow visualization"
        return mock_visualizer

    @pytest.fixture
    def phase2_instance(self, mock_session_data, mock_get_user_input, mock_workflow_visualizer):
        """Create Phase2PlanningReflecting instance with mocked dependencies."""
        return Phase2PlanningReflecting(
            session_data=mock_session_data,
            get_user_input_func=mock_get_user_input,
            workflow_visualizer=mock_workflow_visualizer
        )

    def test_extract_workflow_description_from_plain_text(self, phase2_instance):
        """Test extracting workflow description from plain text input."""
        plain_text = "Create a workflow that processes data and sends notifications"

        result = phase2_instance._extract_workflow_description_from_input(plain_text)

        assert result == plain_text

    def test_extract_workflow_description_from_json_with_description_key(self, phase2_instance):
        """Test extracting workflow description from JSON with description key."""
        json_input = json.dumps({
            "workflow_description": "Process user data and generate reports",
            "other_field": "ignored"
        })

        result = phase2_instance._extract_workflow_description_from_input(json_input)

        assert result == "Process user data and generate reports"

    def test_extract_workflow_description_from_json_with_description_key_alt(self, phase2_instance):
        """Test extracting workflow description from JSON with 'description' key."""
        json_input = json.dumps({
            "description": "Automated testing workflow",
            "version": "1.0"
        })

        result = phase2_instance._extract_workflow_description_from_input(json_input)

        assert result == "Automated testing workflow"

    def test_collect_workflow_description_quit(self, phase2_instance, mock_get_user_input):
        """Test collect_workflow_description when user quits."""
        mock_get_user_input.return_value = "quit"

        result = phase2_instance.collect_workflow_description()

        assert result is None
        mock_get_user_input.assert_called_once()

    def test_collect_user_feedback(self, phase2_instance, mock_get_user_input):
        """Test collect_user_feedback method."""
        feedback = "Add more error handling to the workflow"
        mock_get_user_input.return_value = feedback

        with patch('builtins.print'):
            result = phase2_instance.collect_user_feedback()

        assert result == feedback
        mock_get_user_input.assert_called_once_with("Your feedback:", "text")

    def test_config_planner_with_custom_values(self, phase2_instance, mock_get_user_input):
        """Test config_planner method with custom values."""
        mock_get_user_input.side_effect = [
            "custom-model-id",
            "25",
            "10000"
        ]

        phase2_instance.config_planner()

        assert phase2_instance.model_id == "custom-model-id"
        assert phase2_instance.max_interactions == "25"
        assert phase2_instance.max_tokens == "10000"
        assert mock_get_user_input.call_count == 3

    def test_config_planner_with_default_values(self, phase2_instance, mock_get_user_input):
        """Test config_planner method with default values."""
        mock_get_user_input.side_effect = ["", "", ""]

        phase2_instance.config_planner()

        assert phase2_instance.model_id == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        assert phase2_instance.max_interactions == 20
        assert phase2_instance.max_tokens == 8000
        assert mock_get_user_input.call_count == 3

    def test_process_feedback_approve(self, phase2_instance):
        """Test process_feedback with 'approve' command."""
        with patch('builtins.print'):
            result = phase2_instance.process_feedback("approve")

        assert result == "next"

    def test_process_feedback_back(self, phase2_instance):
        """Test process_feedback with 'back' command."""
        with patch('builtins.print'):
            result = phase2_instance.process_feedback("back")

        assert result == "back"

    def test_process_feedback_restart(self, phase2_instance):
        """Test process_feedback with 'restart' command."""
        with patch('builtins.print'):
            result = phase2_instance.process_feedback("restart")

        assert result == "restart"

    def test_process_feedback_iterate(self, phase2_instance):
        """Test process_feedback with regular feedback (iterate)."""
        feedback = "Please add validation steps"

        with patch.object(phase2_instance, 'reflect_workflow_plan', return_value=True) as mock_reflect:
            result = phase2_instance.process_feedback(feedback)

        assert result == "iterate"
        mock_reflect.assert_called_once_with(feedback)

class TestRunMethod:
    """Test run method - the main orchestration method."""

    @pytest.fixture
    def mock_session_data_with_tools(self):
        """Create mock session data with tools."""
        return {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": [],
                    "returns": {"name": "result", "type": "string", "description": "Test result"}
                }
            ],
            "workflow_plan": {},
            "claude_messages": []
        }

    @pytest.fixture
    def mock_session_data_no_tools(self):
        """Create mock session data without tools."""
        return {
            "tools": [],
            "workflow_plan": {},
            "claude_messages": []
        }

    @pytest.fixture
    def mock_get_user_input(self):
        """Create mock get_user_input function."""
        return Mock()

    @pytest.fixture
    def mock_workflow_visualizer(self):
        """Create mock workflow visualizer."""
        mock_visualizer = Mock()
        mock_visualizer.visualize_workflow.return_value = "Mocked workflow visualization"
        return mock_visualizer

    @pytest.fixture
    def phase2_instance_with_tools(self, mock_session_data_with_tools, mock_get_user_input, mock_workflow_visualizer):
        """Create Phase2PlanningReflecting instance with tools."""
        return Phase2PlanningReflecting(
            session_data=mock_session_data_with_tools,
            get_user_input_func=mock_get_user_input,
            workflow_visualizer=mock_workflow_visualizer
        )

    @pytest.fixture
    def phase2_instance_no_tools(self, mock_session_data_no_tools, mock_get_user_input, mock_workflow_visualizer):
        """Create Phase2PlanningReflecting instance without tools."""
        return Phase2PlanningReflecting(
            session_data=mock_session_data_no_tools,
            get_user_input_func=mock_get_user_input,
            workflow_visualizer=mock_workflow_visualizer
        )

    def test_run_no_tools_available(self, phase2_instance_no_tools):
        """Test run method when no tools are available from Phase 1."""
        with patch.object(phase2_instance_no_tools, 'print_phase_header') as mock_header:
            with patch('builtins.print'):
                result = phase2_instance_no_tools.run()

        assert result == "back"
        mock_header.assert_called_once()

    def test_run_successful_generation_approve_next(self, phase2_instance_with_tools, mock_workflow_visualizer):
        """Test run method with successful generation and user approves to proceed."""
        with patch.object(phase2_instance_with_tools, 'print_phase_header'):
            with patch.object(phase2_instance_with_tools, 'print_workflow_guidance'):
                with patch.object(phase2_instance_with_tools, 'collect_workflow_description', return_value="Create a workflow"):
                    with patch.object(phase2_instance_with_tools, 'config_planner'):
                        with patch.object(phase2_instance_with_tools, 'generate_workflow_plan', return_value=True):
                            with patch.object(phase2_instance_with_tools, 'collect_user_feedback', return_value="approve"):
                                with patch.object(phase2_instance_with_tools, 'process_feedback', return_value="next"):
                                    with patch('builtins.print'):
                                        result = phase2_instance_with_tools.run()

        assert result == "next"
        mock_workflow_visualizer.visualize_workflow.assert_called()

    def test_run_successful_generation_go_back(self, phase2_instance_with_tools, mock_workflow_visualizer):
        """Test run method with successful generation and user chooses to go back."""
        with patch.object(phase2_instance_with_tools, 'print_phase_header'):
            with patch.object(phase2_instance_with_tools, 'print_workflow_guidance'):
                with patch.object(phase2_instance_with_tools, 'collect_workflow_description', return_value="Create a workflow"):
                    with patch.object(phase2_instance_with_tools, 'config_planner'):
                        with patch.object(phase2_instance_with_tools, 'generate_workflow_plan', return_value=True):
                            with patch.object(phase2_instance_with_tools, 'collect_user_feedback', return_value="back"):
                                with patch.object(phase2_instance_with_tools, 'process_feedback', return_value="back"):
                                    with patch('builtins.print'):
                                        result = phase2_instance_with_tools.run()

        assert result == "back"
        mock_workflow_visualizer.visualize_workflow.assert_called()

    def test_run_successful_generation_restart_workflow(self, phase2_instance_with_tools, mock_workflow_visualizer):
        """Test run method with successful generation, user restarts, then quits."""
        with patch.object(phase2_instance_with_tools, 'print_phase_header'):
            with patch.object(phase2_instance_with_tools, 'print_workflow_guidance'):
                with patch.object(phase2_instance_with_tools, 'collect_workflow_description', side_effect=["Create a workflow", None]):
                    with patch.object(phase2_instance_with_tools, 'config_planner'):
                        with patch.object(phase2_instance_with_tools, 'generate_workflow_plan', return_value=True):
                            with patch.object(phase2_instance_with_tools, 'collect_user_feedback', return_value="restart"):
                                with patch.object(phase2_instance_with_tools, 'process_feedback', return_value="restart"):
                                    with patch('builtins.print'):
                                        result = phase2_instance_with_tools.run()

        assert result is None
        mock_workflow_visualizer.visualize_workflow.assert_called()

    def test_run_successful_generation_iterate_then_approve(self, phase2_instance_with_tools, mock_workflow_visualizer):
        """Test run method with successful generation, user iterates, then approves."""
        with patch.object(phase2_instance_with_tools, 'print_phase_header'):
            with patch.object(phase2_instance_with_tools, 'print_workflow_guidance'):
                with patch.object(phase2_instance_with_tools, 'collect_workflow_description', return_value="Create a workflow"):
                    with patch.object(phase2_instance_with_tools, 'config_planner'):
                        with patch.object(phase2_instance_with_tools, 'generate_workflow_plan', return_value=True):
                            with patch.object(phase2_instance_with_tools, 'collect_user_feedback', side_effect=["improve the workflow", "approve"]):
                                with patch.object(phase2_instance_with_tools, 'process_feedback', side_effect=["iterate", "next"]):
                                    with patch('builtins.print'):
                                        result = phase2_instance_with_tools.run()

        assert result == "next"
        assert mock_workflow_visualizer.visualize_workflow.call_count == 2

    def test_run_multiple_iterations_then_restart(self, phase2_instance_with_tools, mock_workflow_visualizer):
        """Test run method with multiple iterations, then restart, then quit."""
        with patch.object(phase2_instance_with_tools, 'print_phase_header'):
            with patch.object(phase2_instance_with_tools, 'print_workflow_guidance'):
                with patch.object(phase2_instance_with_tools, 'collect_workflow_description', side_effect=["Create a workflow", None]):
                    with patch.object(phase2_instance_with_tools, 'config_planner'):
                        with patch.object(phase2_instance_with_tools, 'generate_workflow_plan', return_value=True):
                            with patch.object(phase2_instance_with_tools, 'collect_user_feedback', side_effect=["improve", "add more", "restart"]):
                                with patch.object(phase2_instance_with_tools, 'process_feedback', side_effect=["iterate", "iterate", "restart"]):
                                    with patch('builtins.print'):
                                        result = phase2_instance_with_tools.run()

        assert result is None
        # Should be called multiple times due to iterations
        assert mock_workflow_visualizer.visualize_workflow.call_count >= 2

    def test_run_empty_workflow_plan_visualization(self, phase2_instance_with_tools, mock_workflow_visualizer):
        """Test run method with empty workflow plan in session data."""
        # Ensure workflow_plan is empty
        phase2_instance_with_tools.session_data["workflow_plan"] = {}

        with patch.object(phase2_instance_with_tools, 'print_phase_header'):
            with patch.object(phase2_instance_with_tools, 'print_workflow_guidance'):
                with patch.object(phase2_instance_with_tools, 'collect_workflow_description', return_value="Create a workflow"):
                    with patch.object(phase2_instance_with_tools, 'config_planner'):
                        with patch.object(phase2_instance_with_tools, 'generate_workflow_plan', return_value=True):
                            with patch.object(phase2_instance_with_tools, 'collect_user_feedback', return_value="approve"):
                                with patch.object(phase2_instance_with_tools, 'process_feedback', return_value="next"):
                                    with patch('builtins.print'):
                                        result = phase2_instance_with_tools.run()

        assert result == "next"
        mock_workflow_visualizer.visualize_workflow.assert_called_with({})

    def test_run_tools_count_display(self, phase2_instance_with_tools):
        """Test run method displays correct tools count."""
        # Add multiple tools to session data
        phase2_instance_with_tools.session_data["tools"] = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"},
            {"name": "tool3", "description": "Third tool"}
        ]

        with patch.object(phase2_instance_with_tools, 'print_phase_header'):
            with patch.object(phase2_instance_with_tools, 'print_workflow_guidance'):
                with patch.object(phase2_instance_with_tools, 'collect_workflow_description', return_value="Create a workflow"):
                    with patch.object(phase2_instance_with_tools, 'config_planner'):
                        with patch.object(phase2_instance_with_tools, 'generate_workflow_plan', return_value=True):
                            with patch.object(phase2_instance_with_tools, 'collect_user_feedback', return_value="approve"):
                                with patch.object(phase2_instance_with_tools, 'process_feedback', return_value="next"):
                                    with patch('builtins.print') as mock_print:
                                        result = phase2_instance_with_tools.run()

        assert result == "next"
        # Check that the tools count was printed
        call_args = [str(call) for call in mock_print.call_args_list]
        tools_count_found = any("Found 3 available tools" in arg for arg in call_args)
        assert tools_count_found


class TestPhase2PrintMethods:
    """Test print methods for Phase2PlanningReflecting."""

    @pytest.fixture
    def phase2_instance(self):
        """Create Phase2PlanningReflecting instance for print method testing."""
        return Phase2PlanningReflecting(
            session_data={"tools": [], "claude_messages": []},
            get_user_input_func=Mock(),
            workflow_visualizer=Mock()
        )

    def test_print_phase_header(self, phase2_instance):
        """Test print_phase_header method."""
        with patch('builtins.print') as mock_print:
            phase2_instance.print_phase_header()

        # Verify that print was called multiple times
        assert mock_print.call_count > 0
        # Check that the header contains expected text
        call_args = [str(call) for call in mock_print.call_args_list]
        header_found = any("PHASE 2: PLANNING & USER REFLECTION" in arg for arg in call_args)
        assert header_found

    def test_print_workflow_guidance(self, phase2_instance):
        """Test print_workflow_guidance method."""
        with patch('builtins.print') as mock_print:
            phase2_instance.print_workflow_guidance()

        # Verify that print was called multiple times for the guidance
        assert mock_print.call_count > 0
        # Check that guidance contains expected content
        call_args = [str(call) for call in mock_print.call_args_list]
        guidance_found = any("Workflow Description Input" in arg for arg in call_args)
        assert guidance_found
