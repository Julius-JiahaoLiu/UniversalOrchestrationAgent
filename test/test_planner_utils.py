"""
Tests for Planner Utils

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

from unittest.mock import Mock, patch

from elastic_gumby_universal_orch_agent_prototype.planner.utils import generate_plan, reflect_plan


class TestGeneratePlan:
    """Test generate_plan function."""

    @patch('elastic_gumby_universal_orch_agent_prototype.planner.utils.IterativePlanner')
    def test_generate_plan_success(self, mock_planner_class):
        """Test successful workflow plan generation."""
        # Mock the planner instance and its methods
        mock_planner = Mock()
        mock_planner.workflow_schema = {"type": "object", "properties": {}}
        mock_planner.iterative_planning.return_value = {
            "name": "test_workflow",
            "description": "A test workflow",
            "nodes": []
        }
        mock_planner.claude_messages = [{"role": "assistant", "content": "Generated plan"}]
        mock_planner_class.return_value = mock_planner

        # Test data
        workflow_description = "Create a simple data processing workflow"
        available_tools = [{"name": "data_processor", "description": "Processes data"}]
        model_id = "test-model-id"
        max_interactions = 15
        max_tokens = 6000

        with patch('builtins.print'):
            result_plan, result_messages = generate_plan(
                workflow_description=workflow_description,
                available_tools=available_tools,
                model_id=model_id,
                max_interactions=max_interactions,
                max_tokens=max_tokens
            )

        # Verify planner was created with correct parameters
        mock_planner_class.assert_called_once_with(
            model_id=model_id,
            max_interactions=max_interactions,
            max_tokens=max_tokens
        )

        # Verify iterative_planning was called with correct parameters
        mock_planner.iterative_planning.assert_called_once()
        call_args = mock_planner.iterative_planning.call_args

        # Check that messages contain the workflow description and tools
        messages = call_args[1]['messages']
        assert len(messages) == 1
        assert workflow_description in messages[0]['content']
        assert "data_processor" in messages[0]['content']

        # Check system prompt and tool are provided
        assert 'system_prompt' in call_args[1]
        assert 'workflow_execution_tool' in call_args[1]
        assert 'available_tools' in call_args[1]

        # Verify return values
        assert result_plan["name"] == "test_workflow"
        assert result_messages == mock_planner.claude_messages

    @patch('elastic_gumby_universal_orch_agent_prototype.planner.utils.IterativePlanner')
    def test_generate_plan_default_model(self, mock_planner_class):
        """Test generate_plan with default model ID and parameters."""
        mock_planner = Mock()
        mock_planner.workflow_schema = {}
        mock_planner.iterative_planning.return_value = {"name": "default_workflow"}
        mock_planner.claude_messages = []
        mock_planner_class.return_value = mock_planner

        with patch('builtins.print'):
            generate_plan("Simple workflow", [])

        # Verify default parameters are used
        mock_planner_class.assert_called_once_with(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_interactions=20,
            max_tokens=8000
        )

    @patch('elastic_gumby_universal_orch_agent_prototype.planner.utils.IterativePlanner')
    def test_generate_plan_empty_tools(self, mock_planner_class):
        """Test generate_plan with empty tools list."""
        mock_planner = Mock()
        mock_planner.workflow_schema = {}
        mock_planner.iterative_planning.return_value = {"name": "empty_tools_workflow"}
        mock_planner.claude_messages = []
        mock_planner_class.return_value = mock_planner

        with patch('builtins.print'):
            result_plan, _ = generate_plan("Workflow with no tools", [])

        # Verify it handles empty tools gracefully
        assert result_plan["name"] == "empty_tools_workflow"
        mock_planner.iterative_planning.assert_called_once()
        
        # Verify default parameters are used
        mock_planner_class.assert_called_once_with(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_interactions=20,
            max_tokens=8000
        )


class TestReflectPlan:
    """Test reflect_plan function."""

    @patch('elastic_gumby_universal_orch_agent_prototype.planner.utils.IterativePlanner')
    def test_reflect_plan_success(self, mock_planner_class):
        """Test successful workflow plan reflection."""
        # Mock the planner instance
        mock_planner = Mock()
        mock_planner.workflow_schema = {"type": "object", "properties": {}}
        mock_planner.iterative_planning.return_value = {
            "name": "updated_workflow",
            "description": "An updated workflow",
            "nodes": [{"id": "new_node", "type": "task"}]
        }
        mock_planner.claude_messages = [{"role": "assistant", "content": "Updated plan"}]
        mock_planner_class.return_value = mock_planner

        # Test data
        existing_plan = {
            "name": "original_workflow",
            "description": "Original workflow",
            "nodes": [{"id": "old_node", "type": "task"}]
        }
        user_feedback = "Add error handling to the workflow"
        available_tools = [{"name": "error_handler", "description": "Handles errors"}]
        model_id = "reflection-model-id"
        max_interactions = 10
        max_tokens = 5000

        with patch('builtins.print'):
            result_plan, result_messages = reflect_plan(
                existing_workflow_plan=existing_plan,
                user_feedback=user_feedback,
                available_tools=available_tools,
                model_id=model_id,
                max_interactions=max_interactions,
                max_tokens=max_tokens
            )

        # Verify planner was created with correct parameters
        mock_planner_class.assert_called_once_with(
            model_id=model_id,
            max_interactions=max_interactions,
            max_tokens=max_tokens
        )

        # Verify iterative_planning was called
        mock_planner.iterative_planning.assert_called_once()
        call_args = mock_planner.iterative_planning.call_args

        # Check that messages contain the existing plan, feedback, and tools
        messages = call_args[1]['messages']
        assert len(messages) == 1
        message_content = messages[0]['content']
        assert "original_workflow" in message_content
        assert user_feedback in message_content
        assert "error_handler" in message_content

        # Check system prompt and tool are provided
        assert 'system_prompt' in call_args[1]
        assert 'workflow_execution_tool' in call_args[1]
        assert 'available_tools' in call_args[1]

        # Verify return values
        assert result_plan["name"] == "updated_workflow"
        assert result_messages == mock_planner.claude_messages

    @patch('elastic_gumby_universal_orch_agent_prototype.planner.utils.IterativePlanner')
    def test_reflect_plan_default_model(self, mock_planner_class):
        """Test reflect_plan with default model ID and parameters."""
        mock_planner = Mock()
        mock_planner.workflow_schema = {}
        mock_planner.iterative_planning.return_value = {"name": "reflected_workflow"}
        mock_planner.claude_messages = []
        mock_planner_class.return_value = mock_planner

        existing_plan = {"name": "test_plan"}
        feedback = "Make improvements"

        with patch('builtins.print'):
            reflect_plan(existing_plan, feedback, [])

        # Verify default parameters are used
        mock_planner_class.assert_called_once_with(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_interactions=20,
            max_tokens=8000
        )

    @patch('elastic_gumby_universal_orch_agent_prototype.planner.utils.IterativePlanner')
    def test_reflect_plan_empty_feedback(self, mock_planner_class):
        """Test reflect_plan with empty feedback."""
        mock_planner = Mock()
        mock_planner.workflow_schema = {}
        mock_planner.iterative_planning.return_value = {"name": "unchanged_workflow"}
        mock_planner.claude_messages = []
        mock_planner_class.return_value = mock_planner

        existing_plan = {"name": "existing_plan"}
        empty_feedback = ""

        with patch('builtins.print'):
            result_plan, _ = reflect_plan(existing_plan, empty_feedback, [])

        # Verify it handles empty feedback gracefully
        assert result_plan["name"] == "unchanged_workflow"
        mock_planner.iterative_planning.assert_called_once()

        # Check that empty feedback is still passed to the planner
        call_args = mock_planner.iterative_planning.call_args
        messages = call_args[1]['messages']
        assert empty_feedback in messages[0]['content']
        
        # Verify default parameters are used
        mock_planner_class.assert_called_once_with(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_interactions=20,
            max_tokens=8000
        )