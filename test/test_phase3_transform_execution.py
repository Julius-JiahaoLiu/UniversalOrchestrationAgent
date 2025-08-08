"""
Tests for Phase3TransformExecution

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from elastic_gumby_universal_orch_agent_prototype.phases.phase3_transform_execution import Phase3TransformExecution

class TestRunMethod:
    """Test run method scenarios."""

    @pytest.fixture
    def mock_session_data(self):
        """Create mock session data."""
        return {
            "tools": [{"name": "test_tool", "description": "Test tool"}],
            "workflow_plan": {"name": "Test Workflow", "steps": []}
        }

    @pytest.fixture
    def mock_get_user_input(self):
        """Create mock get_user_input function."""
        return Mock()

    @pytest.fixture
    def mock_session_dir(self):
        """Create mock session directory."""
        return Path("/tmp/test_session")

    @pytest.fixture
    def phase3_instance(self, mock_session_data, mock_get_user_input, mock_session_dir):
        """Create Phase3TransformExecution instance with mocked dependencies."""
        return Phase3TransformExecution(
            session_data=mock_session_data,
            get_user_input_func=mock_get_user_input,
            session_dir=mock_session_dir
        )

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase3_transform_execution.StateMachineTransformer')
    def test_run_full_workflow_with_deployment(self, mock_transformer_class, phase3_instance, mock_get_user_input):
        """Test run method with full workflow including deployment and execution."""
        mock_transformer = Mock()
        mock_transformer_class.return_value = mock_transformer
        mock_get_user_input.return_value = "y"
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:TestStateMachine"
        
        with patch.object(phase3_instance, 'print_phase_header'), \
             patch.object(phase3_instance, 'deploy_state_machine', return_value=test_arn), \
             patch.object(phase3_instance, 'execute_state_machine') as mock_execute, \
             patch.object(phase3_instance, 'delete_state_machine') as mock_delete, \
             patch.object(phase3_instance, 'handle_post_execution_options', return_value=None) as mock_handle:
            
            result = phase3_instance.run()
            
            assert result is None
            mock_transformer.save_state_machine.assert_called_once()
            mock_execute.assert_called_once_with(test_arn)
            mock_delete.assert_called_once_with(test_arn)
            mock_handle.assert_called_once()


class TestDeployStateMachine:
    """Test deploy_state_machine method."""

    @pytest.fixture
    def mock_session_data(self):
        """Create mock session data."""
        return {
            "workflow_plan": {"name": "Test State Machine", "steps": []}
        }

    @pytest.fixture
    def mock_get_user_input(self):
        """Create mock get_user_input function."""
        return Mock()

    @pytest.fixture
    def mock_session_dir(self, tmp_path):
        """Create temporary session directory with state machine file."""
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        
        # Create mock state machine definition file
        state_machine_file = session_dir / "state_machine.asl.json"
        state_machine_def = {
            "Comment": "Test state machine",
            "StartAt": "HelloWorld",
            "States": {
                "HelloWorld": {
                    "Type": "Pass",
                    "Result": "Hello World!",
                    "End": True
                }
            }
        }
        state_machine_file.write_text(json.dumps(state_machine_def))
        
        return session_dir

    @pytest.fixture
    def phase3_instance(self, mock_session_data, mock_get_user_input, mock_session_dir):
        """Create Phase3TransformExecution instance with mocked dependencies."""
        return Phase3TransformExecution(
            session_data=mock_session_data,
            get_user_input_func=mock_get_user_input,
            session_dir=mock_session_dir
        )

    @patch('boto3.client')
    def test_deploy_state_machine_successful_deployment(self, mock_boto3_client, phase3_instance, mock_get_user_input):
        """Test successful state machine deployment."""
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:Test_State_Machine_v1"
        mock_get_user_input.side_effect = ["y", "arn:aws:iam::123456789012:role/StepFunctionsRole"]
        
        # Mock boto3 client
        mock_sfn_client = Mock()
        mock_sfn_client.create_state_machine.return_value = {"stateMachineArn": test_arn}
        mock_boto3_client.return_value = mock_sfn_client
        
        with patch('builtins.print'):
            result = phase3_instance.deploy_state_machine()
            
        assert result == test_arn
        mock_boto3_client.assert_called_once_with('stepfunctions')
        mock_sfn_client.create_state_machine.assert_called_once()
        
        # Verify the call arguments
        call_args = mock_sfn_client.create_state_machine.call_args
        assert call_args[1]['name'] == "Test_State_Machine_v1"
        assert call_args[1]['roleArn'] == "arn:aws:iam::123456789012:role/StepFunctionsRole"
        assert call_args[1]['type'] == "STANDARD"

    @patch('boto3.client')
    def test_deploy_state_machine_deployment_failure_then_success(self, mock_boto3_client, phase3_instance, mock_get_user_input):
        """Test deployment failure followed by successful retry."""
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:Test_State_Machine_v2"
        mock_get_user_input.side_effect = [
            "y", "arn:aws:iam::123456789012:role/StepFunctionsRole",  # First attempt
            "y", "arn:aws:iam::123456789012:role/StepFunctionsRole"   # Second attempt
        ]
        
        # Mock boto3 client
        mock_sfn_client = Mock()
        # First call fails, second succeeds
        mock_sfn_client.create_state_machine.side_effect = [
            Exception("State machine already exists"),
            {"stateMachineArn": test_arn}
        ]
        mock_boto3_client.return_value = mock_sfn_client
        
        with patch('builtins.print'):
            result = phase3_instance.deploy_state_machine()
            
        assert result == test_arn
        assert mock_sfn_client.create_state_machine.call_count == 2

    def test_deploy_state_machine_user_declines(self, phase3_instance, mock_get_user_input):
        """Test when user declines deployment."""
        mock_get_user_input.return_value = "n"
        
        with patch('builtins.print'):
            result = phase3_instance.deploy_state_machine()
            
        assert result == ""

class TestExecuteStateMachine:
    """Test execute_state_machine method."""

    @pytest.fixture
    def mock_session_dir(self, tmp_path):
        """Create temporary session directory with execution input file."""
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        
        # Create mock execution input file
        exec_input_file = session_dir / "exec_input.json"
        exec_input = {
            "inputParam1": ["value1", "value2"],
            "inputParam2": "static_value",
            "nested": {
                "param": ["option1", "option2"]
            }
        }
        exec_input_file.write_text(json.dumps(exec_input))
        
        return session_dir

    @pytest.fixture
    def phase3_instance(self, mock_session_dir):
        """Create Phase3TransformExecution instance."""
        instance = Phase3TransformExecution(
            session_data={"tools": [], "workflow_plan": {}},
            get_user_input_func=Mock(),
            session_dir=mock_session_dir
        )
        # Mock the SFN client
        instance.SFN_client = Mock()
        return instance

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase3_transform_execution.sleep')
    def test_execute_state_machine_successful_execution(self, mock_sleep, phase3_instance):
        """Test successful state machine execution monitoring."""
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:TestStateMachine"
        execution_arn = "arn:aws:states:us-west-2:123456789012:execution:TestStateMachine:test-execution"
        
        phase3_instance.get_user_input.side_effect = ["y", "n"]
        
        # Mock start execution response
        phase3_instance.SFN_client.start_execution.return_value = {"executionArn": execution_arn}
        
        # Mock execution history response - successful execution
        phase3_instance.SFN_client.get_execution_history.return_value = {
            "events": [
                {
                    "id": 1,
                    "type": "ExecutionStarted"
                },
                {
                    "id": 2,
                    "type": "ExecutionSucceeded"
                }
            ]
        }
        
        with patch('builtins.print'), \
             patch.object(phase3_instance, '_random_choose_execution_input', return_value={"test": "input"}):
            
            phase3_instance.execute_state_machine(test_arn)
            
        phase3_instance.SFN_client.start_execution.assert_called_once()
        phase3_instance.SFN_client.get_execution_history.assert_called()

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase3_transform_execution.sleep')
    def test_execute_state_machine_failed_execution(self, mock_sleep, phase3_instance):
        """Test failed state machine execution monitoring."""
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:TestStateMachine"
        execution_arn = "arn:aws:states:us-west-2:123456789012:execution:TestStateMachine:test-execution"
        
        phase3_instance.get_user_input.side_effect = ["y", "n"]
        
        # Mock start execution response
        phase3_instance.SFN_client.start_execution.return_value = {"executionArn": execution_arn}
        
        # Mock execution history response - failed execution
        phase3_instance.SFN_client.get_execution_history.return_value = {
            "events": [
                {
                    "id": 1,
                    "type": "ExecutionStarted"
                },
                {
                    "id": 2,
                    "type": "ExecutionFailed",
                    "executionFailedEventDetails": {
                        "cause": '{"errorType": "States.TaskFailed", "errorMessage": "Task failed"}'
                    }
                }
            ]
        }
        
        with patch('builtins.print'), \
             patch.object(phase3_instance, '_random_choose_execution_input', return_value={"test": "input"}):
            
            phase3_instance.execute_state_machine(test_arn)
            
        phase3_instance.SFN_client.start_execution.assert_called_once()
        phase3_instance.SFN_client.get_execution_history.assert_called()

    def test_execute_state_machine_start_execution_error(self, phase3_instance):
        """Test error handling when starting execution fails."""
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:TestStateMachine"
        
        phase3_instance.get_user_input.side_effect = ["y", "n"]  # Try once, then stop
        
        # Mock boto3 client error
        phase3_instance.SFN_client.start_execution.side_effect = Exception("Invalid state machine")
        
        with patch('builtins.print'), \
             patch.object(phase3_instance, '_random_choose_execution_input', return_value={"test": "input"}):
            
            phase3_instance.execute_state_machine(test_arn)
            
        phase3_instance.SFN_client.start_execution.assert_called_once()

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase3_transform_execution.sleep')
    def test_execute_state_machine_history_error(self, mock_sleep, phase3_instance):
        """Test error handling when getting execution history fails."""
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:TestStateMachine"
        execution_arn = "arn:aws:states:us-west-2:123456789012:execution:TestStateMachine:test-execution"
        
        phase3_instance.get_user_input.side_effect = ["y", "n"]
        
        # Mock successful start, failed history
        phase3_instance.SFN_client.start_execution.return_value = {"executionArn": execution_arn}
        phase3_instance.SFN_client.get_execution_history.side_effect = Exception("Access denied")
        
        with patch('builtins.print'), \
             patch.object(phase3_instance, '_random_choose_execution_input', return_value={"test": "input"}):
            
            phase3_instance.execute_state_machine(test_arn)
            
        phase3_instance.SFN_client.start_execution.assert_called_once()
        phase3_instance.SFN_client.get_execution_history.assert_called_once()

    def test_execute_state_machine_user_declines(self, phase3_instance):
        """Test when user declines execution."""
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:TestStateMachine"
        
        phase3_instance.get_user_input.return_value = "n"
        
        with patch('builtins.print'):
            phase3_instance.execute_state_machine(test_arn)
            
        phase3_instance.SFN_client.start_execution.assert_not_called()

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase3_transform_execution.sleep')
    def test_execute_state_machine_keyboard_interrupt(self, mock_sleep, phase3_instance):
        """Test KeyboardInterrupt handling during execution monitoring."""
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:TestStateMachine"
        execution_arn = "arn:aws:states:us-west-2:123456789012:execution:TestStateMachine:test-execution"
        
        phase3_instance.get_user_input.side_effect = ["y", "n"]
        
        # Mock start execution response
        phase3_instance.SFN_client.start_execution.return_value = {"executionArn": execution_arn}
        
        # Mock get_execution_history to raise KeyboardInterrupt on first call
        phase3_instance.SFN_client.get_execution_history.side_effect = KeyboardInterrupt()
        
        with patch('builtins.print') as mock_print, \
             patch.object(phase3_instance, '_random_choose_execution_input', return_value={"test": "input"}):
            
            # The method should handle KeyboardInterrupt gracefully
            phase3_instance.execute_state_machine(test_arn)
            
        # Verify execution was started
        phase3_instance.SFN_client.start_execution.assert_called_once()
        
        # Verify execution history was attempted to be checked
        phase3_instance.SFN_client.get_execution_history.assert_called_once()
        
        # Verify stop_execution was called due to KeyboardInterrupt
        phase3_instance.SFN_client.stop_execution.assert_called_once_with(
            executionArn=execution_arn,
            error="UserStoppedExecution",
            cause="Execution interrupted by user"
        )
        
        # Verify the interrupt message was printed (check for core message content)
        interrupt_message_found = False
        for call_args in mock_print.call_args_list:
            if "Execution interrupted by user" in str(call_args):
                interrupt_message_found = True
                break
        assert interrupt_message_found, "Expected interrupt message was not printed"
        
        # Verify sleep was called at least once (before the interrupt)
        mock_sleep.assert_called()


class TestRandomChooseExecutionInput:
    """Test _random_choose_execution_input method."""

    @pytest.fixture
    def phase3_instance(self):
        """Create Phase3TransformExecution instance."""
        return Phase3TransformExecution(
            session_data={},
            get_user_input_func=Mock(),
            session_dir=Path("/tmp/test")
        )

    def test_random_choose_execution_input_complex_nested(self, phase3_instance):
        """Test _random_choose_execution_input with complex nested structure."""
        input_data = {
            "level1": {
                "level2": {
                    "choices": ["deep1", "deep2"],
                    "static": "deep_value"
                },
                "list_at_level2": ["mid1", "mid2"]
            },
            "top_choices": ["top1", "top2", "top3"]
        }
        
        result = phase3_instance._random_choose_execution_input(input_data)
        
        assert result["level1"]["level2"]["choices"] in ["deep1", "deep2"]
        assert result["level1"]["level2"]["static"] == "deep_value"
        assert result["level1"]["list_at_level2"] in ["mid1", "mid2"]

class TestDeleteStateMachine:
    """Test delete_state_machine method."""

    @pytest.fixture
    def phase3_instance(self):
        """Create Phase3TransformExecution instance."""
        instance = Phase3TransformExecution(
            session_data={},
            get_user_input_func=Mock(),
            session_dir=Path("/tmp/test")
        )
        # Mock the SFN client
        instance.SFN_client = Mock()
        return instance

    def test_delete_state_machine_successful_deletion(self, phase3_instance):
        """Test successful state machine deletion."""
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:TestStateMachine"
        phase3_instance.get_user_input.return_value = "y"
        
        with patch('builtins.print'):
            phase3_instance.delete_state_machine(test_arn)
            
        phase3_instance.SFN_client.delete_state_machine.assert_called_once_with(stateMachineArn=test_arn)

    def test_delete_state_machine_deletion_error(self, phase3_instance):
        """Test error handling during state machine deletion."""
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:TestStateMachine"
        phase3_instance.get_user_input.return_value = "y"
        
        # Mock boto3 client error
        phase3_instance.SFN_client.delete_state_machine.side_effect = Exception("State machine not found")
        
        with patch('builtins.print'):
            phase3_instance.delete_state_machine(test_arn)
            
        phase3_instance.SFN_client.delete_state_machine.assert_called_once_with(stateMachineArn=test_arn)

    def test_delete_state_machine_user_declines(self, phase3_instance):
        """Test when user declines deletion."""
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:TestStateMachine"
        phase3_instance.get_user_input.return_value = "n"
        
        with patch('builtins.print'):
            phase3_instance.delete_state_machine(test_arn)
            
        phase3_instance.SFN_client.delete_state_machine.assert_not_called()


class TestHandlePostExecutionOptions:
    """Test handle_post_execution_options method."""

    @pytest.fixture
    def phase3_instance(self):
        """Create Phase3TransformExecution instance."""
        return Phase3TransformExecution(
            session_data={},
            get_user_input_func=Mock(),
            session_dir=Path("/tmp/test")
        )

    def test_handle_post_execution_options_invalid_choice_defaults_to_exit(self, phase3_instance):
        """Test handle_post_execution_options with invalid choice defaults to exit."""
        phase3_instance.get_user_input.return_value = "invalid"
        
        with patch('builtins.print'):
            result = phase3_instance.handle_post_execution_options()
            
        assert result is None
        phase3_instance.get_user_input.assert_called_once()

    def test_handle_post_execution_options_empty_choice_defaults_to_exit(self, phase3_instance):
        """Test handle_post_execution_options with empty choice defaults to exit."""
        phase3_instance.get_user_input.return_value = ""
        
        with patch('builtins.print'):
            result = phase3_instance.handle_post_execution_options()
            
        assert result is None
        phase3_instance.get_user_input.assert_called_once()


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple methods."""

    @pytest.fixture
    def mock_session_data(self):
        """Create comprehensive mock session data."""
        return {
            "tools": [
                {"name": "tool1", "description": "First tool"},
                {"name": "tool2", "description": "Second tool"}
            ],
            "workflow_plan": {
                "name": "Integration Test Workflow",
                "steps": [
                    {"name": "step1", "tool": "tool1"},
                    {"name": "step2", "tool": "tool2"}
                ]
            }
        }

    @pytest.fixture
    def mock_session_dir(self, tmp_path):
        """Create temporary session directory with all required files."""
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        
        # Create state machine definition file
        state_machine_file = session_dir / "state_machine.asl.json"
        state_machine_def = {
            "Comment": "Integration test state machine",
            "StartAt": "Step1",
            "States": {
                "Step1": {
                    "Type": "Task",
                    "Resource": "arn:aws:lambda:us-west-2:123456789012:function:tool1",
                    "Next": "Step2"
                },
                "Step2": {
                    "Type": "Task",
                    "Resource": "arn:aws:lambda:us-west-2:123456789012:function:tool2",
                    "End": True
                }
            }
        }
        state_machine_file.write_text(json.dumps(state_machine_def))
        
        # Create execution input file
        exec_input_file = session_dir / "exec_input.json"
        exec_input = {
            "workflow_input": ["test1", "test2"],
            "config": {
                "timeout": 300,
                "retry_count": [1, 2, 3]
            }
        }
        exec_input_file.write_text(json.dumps(exec_input))
        
        return session_dir

    @pytest.fixture
    def phase3_instance(self, mock_session_data, mock_session_dir):
        """Create Phase3TransformExecution instance for integration testing."""
        return Phase3TransformExecution(
            session_data=mock_session_data,
            get_user_input_func=Mock(),
            session_dir=mock_session_dir
        )

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase3_transform_execution.StateMachineTransformer')
    @patch('boto3.client')
    def test_full_integration_successful_workflow(self, mock_boto3_client, mock_transformer_class, phase3_instance):
        """Test full integration scenario with successful workflow execution."""
        # Setup mocks
        mock_transformer = Mock()
        mock_transformer_class.return_value = mock_transformer
        
        test_arn = "arn:aws:states:us-west-2:123456789012:stateMachine:Integration_Test_Workflow_v1"
        execution_arn = "arn:aws:states:us-west-2:123456789012:execution:Integration_Test_Workflow:exec1"
        
        # Mock user inputs: transform=yes, deploy=yes, execute=yes, delete=yes, exit
        phase3_instance.get_user_input.side_effect = [
            "y",  # Transform workflow
            "y", "arn:aws:iam::123456789012:role/StepFunctionsRole",  # Deploy
            "y", "n",  # Execute once, then stop
            "y",  # Delete
            "3"   # Exit
        ]
        
        # Mock boto3 client
        mock_sfn_client = Mock()
        mock_boto3_client.return_value = mock_sfn_client
        
        # Mock boto3 responses
        mock_sfn_client.create_state_machine.return_value = {"stateMachineArn": test_arn}
        mock_sfn_client.start_execution.return_value = {"executionArn": execution_arn}
        mock_sfn_client.get_execution_history.return_value = {
            "events": [
                {"id": 1, "type": "ExecutionStarted"},
                {"id": 2, "type": "ExecutionSucceeded"}
            ]
        }
        
        with patch('builtins.print'), \
             patch('elastic_gumby_universal_orch_agent_prototype.phases.phase3_transform_execution.sleep'):
            
            result = phase3_instance.run()
            
        # Verify the full workflow executed
        assert result is None  # Exit
        mock_transformer.save_state_machine.assert_called_once()
        mock_sfn_client.create_state_machine.assert_called_once()
        mock_sfn_client.start_execution.assert_called_once()
        mock_sfn_client.get_execution_history.assert_called()
        mock_sfn_client.delete_state_machine.assert_called_once()

    def test_integration_user_skips_all_steps(self, phase3_instance):
        """Test integration scenario where user skips all steps."""
        # User declines transformation
        phase3_instance.get_user_input.side_effect = ["n", "3"]  # No transform, exit
        
        with patch('builtins.print'):
            result = phase3_instance.run()
            
        assert result is None  # Exit
        # Only called twice: transform question and post-execution choice
        assert phase3_instance.get_user_input.call_count == 2

    @patch('elastic_gumby_universal_orch_agent_prototype.phases.phase3_transform_execution.StateMachineTransformer')
    @patch('boto3.client')
    def test_integration_transform_only_skip_deployment(self, mock_boto3_client, mock_transformer_class, phase3_instance):
        """Test integration scenario where user transforms but skips deployment."""
        # Setup mocks
        mock_transformer = Mock()
        mock_transformer_class.return_value = mock_transformer
        
        # Mock user inputs: transform=yes, deploy=no, exit
        phase3_instance.get_user_input.side_effect = [
            "y",  # Transform workflow
            "n",  # Skip deployment
            "3"   # Exit
        ]
        
        with patch('builtins.print'):
            result = phase3_instance.run()
            
        # Verify transformation happened but no deployment
        assert result is None  # Exit
        mock_transformer.save_state_machine.assert_called_once()
        mock_boto3_client.assert_not_called()  # No boto3 client created
