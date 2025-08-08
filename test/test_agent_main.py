"""
Tests for AgentMainInterface

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

import json
from unittest.mock import Mock, patch, mock_open

import pytest

from elastic_gumby_universal_orch_agent_prototype.agent_main import AgentMainInterface


class TestAgentMainInterfaceInitialization:
    """Test AgentMainInterface initialization."""

    @patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Path.mkdir')
    @patch('elastic_gumby_universal_orch_agent_prototype.agent_main.ToolsVisualizer')
    @patch('elastic_gumby_universal_orch_agent_prototype.agent_main.WorkflowVisualizer')
    @patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase1ToolsOnboarding')
    @patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase2PlanningReflecting')
    @patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase3TransformExecution')
    @patch('builtins.print')
    def test_initialization_success(self, mock_print, mock_phase3, mock_phase2, mock_phase1, 
                                  mock_workflow_viz, mock_tools_viz, mock_mkdir):
        """Test successful initialization of AgentMainInterface."""
        # Create instance
        agent = AgentMainInterface()
        
        # Verify initial state
        assert agent.current_phase == 1
        assert agent.session_id.startswith("utoa_")
        assert "session_id" in agent.session_data
        assert "created_at" in agent.session_data
        assert agent.session_data["current_phase"] == 1
        assert agent.session_data["phase_history"] == []
        assert agent.session_data["tools"] == []
        assert agent.session_data["claude_messages"] == []
        
        # Verify session directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Verify visualizers created
        mock_tools_viz.assert_called_once()
        mock_workflow_viz.assert_called_once()
        
        # Verify phase handlers created
        mock_phase1.assert_called_once()
        mock_phase2.assert_called_once()
        mock_phase3.assert_called_once()
        
        # Verify initialization message printed
        mock_print.assert_called()

    def test_generate_session_id_format(self):
        """Test session ID generation format."""
        with patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Path.mkdir'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.ToolsVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.WorkflowVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase1ToolsOnboarding'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase2PlanningReflecting'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase3TransformExecution'), \
             patch('builtins.print'):
            
            agent = AgentMainInterface()
            session_id = agent._generate_session_id()
            
            # Verify format: utoa_YYYYMMDD_HHMMSS
            assert session_id.startswith("utoa_")
            assert len(session_id) == 20  # utoa_ + 8 digits + _ + 6 digits
            
            # Verify it contains valid datetime components
            date_time_part = session_id[5:]  # Remove "utoa_"
            assert len(date_time_part) == 15  # YYYYMMDD_HHMMSS
            assert date_time_part[8] == "_"  # Separator

class TestAgentMainInterfaceUserInput:
    """Test user input methods of AgentMainInterface."""

    @pytest.fixture
    def agent_instance(self):
        """Create AgentMainInterface instance with mocked dependencies."""
        with patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Path.mkdir'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.ToolsVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.WorkflowVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase1ToolsOnboarding'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase2PlanningReflecting'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase3TransformExecution'), \
             patch('builtins.print'):
            return AgentMainInterface()

    @patch.object(AgentMainInterface, '_editor')
    @patch('builtins.print')
    def test_get_user_input_text_type(self, mock_print, mock_editor, agent_instance):
        """Test _get_user_input with text input type."""
        mock_editor.return_value = "test input"
        
        result = agent_instance._get_user_input("Enter text:", "text")
        
        assert result == "test input"
        mock_editor.assert_called_once_with(multiline=False)

    @patch.object(AgentMainInterface, '_editor')
    @patch('builtins.print')
    def test_get_user_input_multiline_type(self, mock_print, mock_editor, agent_instance):
        """Test _get_user_input with multiline input type."""
        mock_editor.return_value = "line1\nline2"
        
        result = agent_instance._get_user_input("Enter multiline:", "multiline")
        
        assert result == "line1\nline2"
        mock_editor.assert_called_once_with(multiline=True)

    @patch.object(AgentMainInterface, '_editor')
    @patch('builtins.print')
    def test_get_user_input_file_type_quit(self, mock_print, mock_editor, agent_instance):
        """Test _get_user_input with file input type - quit."""
        mock_editor.return_value = "quit"
        
        result = agent_instance._get_user_input("Enter file content:", "file")
        
        assert result == "quit"

    @patch.object(AgentMainInterface, '_editor')
    @patch('builtins.open', new_callable=mock_open, read_data="file content")
    @patch('builtins.print')
    def test_get_user_input_file_type_file_path(self, mock_print, mock_file, mock_editor, agent_instance):
        """Test _get_user_input with file input type - file path."""
        mock_editor.return_value = "file:test.txt"
        
        result = agent_instance._get_user_input("Enter file content:", "file")
        
        assert result == "file content"
        mock_file.assert_called_once()

    @patch.object(AgentMainInterface, '_editor')
    @patch('builtins.open', side_effect=Exception("File not found"))
    @patch('builtins.print')
    def test_get_user_input_file_type_file_error(self, mock_print, mock_file, mock_editor, agent_instance):
        """Test _get_user_input with file input type - file error."""
        # First call returns file path, second call returns quit to exit recursion
        mock_editor.side_effect = ["file:nonexistent.txt", "quit"]
        
        result = agent_instance._get_user_input("Enter file content:", "file")
        
        # Should return "quit" after file error
        assert result == "quit"
        
        # Verify error message was printed
        error_calls = [call for call in mock_print.call_args_list 
                      if call[0] and "Error reading file" in str(call[0][0])]
        assert len(error_calls) > 0


class TestAgentMainInterfaceSessionManagement:
    """Test session management methods of AgentMainInterface."""

    @pytest.fixture
    def agent_instance(self):
        """Create AgentMainInterface instance with mocked dependencies."""
        with patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Path.mkdir'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.ToolsVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.WorkflowVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase1ToolsOnboarding'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase2PlanningReflecting'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase3TransformExecution'), \
             patch('builtins.print'):
            return AgentMainInterface()

    def test_record_phase_transition(self, agent_instance):
        """Test _record_phase_transition method."""
        agent_instance._record_phase_transition(1, 2, "Test transition")
        
        assert len(agent_instance.session_data["phase_history"]) == 1
        transition = agent_instance.session_data["phase_history"][0]
        
        assert transition["from_phase"] == 1
        assert transition["to_phase"] == 2
        assert transition["reason"] == "Test transition"
        assert "timestamp" in transition

    @patch('builtins.open', new_callable=mock_open)
    @patch.object(AgentMainInterface, '_save_visualization')
    @patch.object(AgentMainInterface, '_save_claude_messages')
    @patch('builtins.print')
    def test_save_session_data_success(self, mock_print, mock_save_claude, mock_save_viz, 
                                     mock_file, agent_instance):
        """Test _save_session_data method success."""
        agent_instance._save_session_data()
        
        # Verify helper methods called
        mock_save_viz.assert_called_once()
        mock_save_claude.assert_called_once()
        
        # Verify file operations
        mock_file.assert_called_once()
        
        # Verify session data updated
        assert "last_updated" in agent_instance.session_data

    @patch('builtins.open', side_effect=Exception("File error"))
    @patch.object(AgentMainInterface, '_save_visualization')
    @patch.object(AgentMainInterface, '_save_claude_messages')
    @patch('builtins.print')
    def test_save_session_data_error(self, mock_print, mock_save_claude, mock_save_viz, 
                                   mock_file, agent_instance):
        """Test _save_session_data method with error."""
        agent_instance._save_session_data()
        
        # Verify error message printed
        error_calls = [call for call in mock_print.call_args_list 
                      if call[0] and "Error saving session data" in str(call[0][0])]
        assert len(error_calls) > 0


class TestAgentMainInterfaceVisualizationSaving:
    """Test visualization saving methods of AgentMainInterface."""

    @pytest.fixture
    def agent_instance(self):
        """Create AgentMainInterface instance with mocked dependencies."""
        with patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Path.mkdir'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.ToolsVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.WorkflowVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase1ToolsOnboarding'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase2PlanningReflecting'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase3TransformExecution'), \
             patch('builtins.print'):
            return AgentMainInterface()

    @patch('builtins.print')
    def test_save_visualization_no_data(self, mock_print, agent_instance):
        """Test _save_visualization with no tools or workflow data."""
        agent_instance._save_visualization()
        
        # Verify warning messages printed
        warning_calls = [call for call in mock_print.call_args_list 
                        if call[0] and "No tools to save" in str(call[0][0])]
        assert len(warning_calls) > 0
        
        warning_calls = [call for call in mock_print.call_args_list 
                        if call[0] and "No workflow plan to save" in str(call[0][0])]
        assert len(warning_calls) > 0

    @patch('builtins.open', side_effect=Exception("File error"))
    @patch('builtins.print')
    def test_save_visualization_error(self, mock_print, mock_file, agent_instance):
        """Test _save_visualization with file error."""
        agent_instance.session_data["tools"] = [{"name": "tool1"}]
        
        agent_instance._save_visualization()
        
        # Verify error message printed
        error_calls = [call for call in mock_print.call_args_list 
                      if call[0] and "Error saving visualizations" in str(call[0][0])]
        assert len(error_calls) > 0


class TestAgentMainInterfaceClaudeMessagesSaving:
    """Test Claude messages saving methods of AgentMainInterface."""

    @pytest.fixture
    def agent_instance(self):
        """Create AgentMainInterface instance with mocked dependencies."""
        with patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Path.mkdir'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.ToolsVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.WorkflowVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase1ToolsOnboarding'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase2PlanningReflecting'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase3TransformExecution'), \
             patch('builtins.print'):
            return AgentMainInterface()

    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')
    def test_save_claude_messages_with_data(self, mock_print, mock_file, agent_instance):
        """Test _save_claude_messages with Claude messages data."""
        # Setup test data
        agent_instance.session_data["claude_messages"] = [
            {"interaction_count": 5, "messages": ["msg1", "msg2"]},
            {"interaction_count": 3, "messages": ["msg3"]}
        ]
        
        agent_instance._save_claude_messages()
        
        # Verify file operations
        mock_file.assert_called_once()
        
        # Verify JSON dump was called - get the written content from all write calls
        handle = mock_file.return_value
        written_content = ''.join(call[0][0] for call in handle.write.call_args_list)
        
        # Parse the JSON content
        written_data = json.loads(written_content)
        assert written_data["session_id"] == agent_instance.session_id
        assert written_data["total_conversations"] == 2
        assert written_data["total_interactions"] == 8  # 5 + 3
        assert len(written_data["raw_messages"]) == 2
        
        # Verify claude_messages removed from session_data
        assert "claude_messages" not in agent_instance.session_data

    @patch('builtins.print')
    def test_save_claude_messages_no_data(self, mock_print, agent_instance):
        """Test _save_claude_messages with no Claude messages."""
        agent_instance._save_claude_messages()
        
        # Verify warning message printed
        warning_calls = [call for call in mock_print.call_args_list 
                        if call[0] and "No Reasoning history to save" in str(call[0][0])]
        assert len(warning_calls) > 0

    @patch('builtins.open', side_effect=Exception("File error"))
    @patch('builtins.print')
    def test_save_claude_messages_error(self, mock_print, mock_file, agent_instance):
        """Test _save_claude_messages with file error."""
        agent_instance.session_data["claude_messages"] = [{"interaction_count": 1}]
        
        agent_instance._save_claude_messages()
        
        # Verify error message printed
        error_calls = [call for call in mock_print.call_args_list 
                      if call[0] and "Could not save reasoning history" in str(call[0][0])]
        assert len(error_calls) > 0


class TestAgentMainInterfaceRunMethod:
    """Test main run method of AgentMainInterface."""

    @pytest.fixture
    def agent_instance(self):
        """Create AgentMainInterface instance with mocked dependencies."""
        with patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Path.mkdir'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.ToolsVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.WorkflowVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase1ToolsOnboarding'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase2PlanningReflecting'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase3TransformExecution'), \
             patch('builtins.print'):
            return AgentMainInterface()

    @patch.object(AgentMainInterface, '_print_banner')
    @patch.object(AgentMainInterface, '_save_session_data')
    @patch.object(AgentMainInterface, '_print_farewell')
    def test_run_phase1_to_phase2_to_phase3_complete(self, mock_farewell, mock_save, 
                                                    mock_banner, agent_instance):
        """Test run method with complete workflow from phase 1 to 3."""
        # Mock phase handlers
        agent_instance.phase1_handler.run = Mock(return_value=True)
        agent_instance.phase2_handler.run = Mock(return_value="next")
        agent_instance.phase3_handler.run = Mock(return_value="complete")
        
        agent_instance.run()
        
        # Verify banner printed
        mock_banner.assert_called_once()
        
        # Verify all phases executed
        agent_instance.phase1_handler.run.assert_called_once()
        agent_instance.phase2_handler.run.assert_called_once()
        agent_instance.phase3_handler.run.assert_called_once()
        
        # Verify final phase is 3
        assert agent_instance.current_phase == 3
        
        # Verify phase transitions recorded
        assert len(agent_instance.session_data["phase_history"]) == 2
        
        # Verify cleanup methods called
        mock_save.assert_called_once()
        mock_farewell.assert_called_once()

    @patch.object(AgentMainInterface, '_print_banner')
    @patch.object(AgentMainInterface, '_save_session_data')
    @patch.object(AgentMainInterface, '_print_farewell')
    def test_run_phase1_exit(self, mock_farewell, mock_save, mock_banner, agent_instance):
        """Test run method with exit in phase 1."""
        # Mock phase 1 to return False (exit)
        agent_instance.phase1_handler.run = Mock(return_value=False)
        
        agent_instance.run()
        
        # Verify only phase 1 executed
        agent_instance.phase1_handler.run.assert_called_once()
        
        # Verify still in phase 1
        assert agent_instance.current_phase == 1
        
        # Verify no phase transitions
        assert len(agent_instance.session_data["phase_history"]) == 0
        
        # Verify cleanup methods called
        mock_save.assert_called_once()
        mock_farewell.assert_called_once()

    @patch.object(AgentMainInterface, '_print_banner')
    @patch.object(AgentMainInterface, '_save_session_data')
    @patch.object(AgentMainInterface, '_print_farewell')
    def test_run_phase3_back_to_phase2(self, mock_farewell, mock_save, mock_banner, agent_instance):
        """Test run method with phase 3 going back to phase 2."""
        # Mock phase handlers
        agent_instance.phase1_handler.run = Mock(return_value=True)
        agent_instance.phase2_handler.run = Mock(side_effect=["next", "exit"])
        agent_instance.phase3_handler.run = Mock(return_value="back")
        
        agent_instance.run()
        
        # Verify execution flow
        agent_instance.phase1_handler.run.assert_called_once()
        assert agent_instance.phase2_handler.run.call_count == 2
        agent_instance.phase3_handler.run.assert_called_once()
        
        # Verify phase transitions recorded
        assert len(agent_instance.session_data["phase_history"]) == 3

    @patch.object(AgentMainInterface, '_print_banner')
    @patch.object(AgentMainInterface, '_save_session_data')
    @patch.object(AgentMainInterface, '_print_farewell')
    def test_run_phase3_restart_to_phase1(self, mock_farewell, mock_save, mock_banner, agent_instance):
        """Test run method with phase 3 restarting to phase 1."""
        # Mock phase handlers
        agent_instance.phase1_handler.run = Mock(side_effect=[True, False])
        agent_instance.phase2_handler.run = Mock(return_value="next")
        agent_instance.phase3_handler.run = Mock(return_value="restart")
        
        agent_instance.run()
        
        # Verify execution flow
        assert agent_instance.phase1_handler.run.call_count == 2
        agent_instance.phase2_handler.run.assert_called_once()
        agent_instance.phase3_handler.run.assert_called_once()
        
        # Verify phase transitions recorded
        assert len(agent_instance.session_data["phase_history"]) == 3

    @patch.object(AgentMainInterface, '_print_banner')
    @patch.object(AgentMainInterface, '_save_session_data')
    @patch.object(AgentMainInterface, '_print_farewell')
    @patch('builtins.print')
    def test_run_keyboard_interrupt(self, mock_print, mock_farewell, mock_save, mock_banner, agent_instance):
        """Test run method with keyboard interrupt."""
        # Mock phase 1 to raise KeyboardInterrupt
        agent_instance.phase1_handler.run = Mock(side_effect=KeyboardInterrupt())
        
        agent_instance.run()
        
        # Verify interrupt message printed
        interrupt_calls = [call for call in mock_print.call_args_list 
                          if call[0] and "Process interrupted by user" in str(call[0][0])]
        assert len(interrupt_calls) > 0
        
        # Verify cleanup methods still called
        mock_save.assert_called_once()
        mock_farewell.assert_called_once()

    @patch.object(AgentMainInterface, '_print_banner')
    @patch.object(AgentMainInterface, '_save_session_data')
    @patch.object(AgentMainInterface, '_print_farewell')
    @patch('builtins.print')
    def test_run_unexpected_error(self, mock_print, mock_farewell, mock_save, mock_banner, agent_instance):
        """Test run method with unexpected error."""
        # Mock phase 1 to raise unexpected exception
        agent_instance.phase1_handler.run = Mock(side_effect=Exception("Unexpected error"))
        
        agent_instance.run()
        
        # Verify error message printed
        error_calls = [call for call in mock_print.call_args_list 
                      if call[0] and "Unexpected error" in str(call[0][0])]
        assert len(error_calls) > 0
        
        # Verify cleanup methods still called
        mock_save.assert_called_once()
        mock_farewell.assert_called_once()


class TestAgentMainInterfaceIntegration:
    """Integration tests for AgentMainInterface combining multiple methods."""

    @pytest.fixture
    def agent_instance(self):
        """Create AgentMainInterface instance with mocked dependencies."""
        with patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Path.mkdir'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.ToolsVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.WorkflowVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase1ToolsOnboarding'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase2PlanningReflecting'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase3TransformExecution'), \
             patch('builtins.print'):
            return AgentMainInterface()

    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')
    def test_complete_session_workflow(self, mock_print, mock_file, agent_instance):
        """Test complete session workflow with data saving."""
        # Setup test data
        agent_instance.session_data["tools"] = [{"name": "test_tool"}]
        agent_instance.session_data["workflow_plan"] = {"steps": ["step1"]}
        agent_instance.session_data["claude_messages"] = [{"interaction_count": 2}]
        
        # Mock visualizer methods
        agent_instance.tools_visualizer.save_tools_visualization = Mock()
        agent_instance.workflow_visualizer.save_workflow_visualization = Mock()
        
        # Record some phase transitions
        agent_instance._record_phase_transition(1, 2, "Phase 1 completed")
        agent_instance._record_phase_transition(2, 3, "Phase 2 completed")
        
        # Save session data
        agent_instance._save_session_data()
        
        # Verify all components worked together
        assert len(agent_instance.session_data["phase_history"]) == 2
        assert "last_updated" in agent_instance.session_data
        
        # Verify file operations occurred
        assert mock_file.call_count >= 3  # session_data.json, tools.json, workflow.json, claude_messages.json
        
        # Verify visualizers called
        agent_instance.tools_visualizer.save_tools_visualization.assert_called_once()
        agent_instance.workflow_visualizer.save_workflow_visualization.assert_called_once()

    def test_session_id_format_consistency(self):
        """Test that session IDs are properly formatted and contain expected components."""
        with patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Path.mkdir'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.ToolsVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.WorkflowVisualizer'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase1ToolsOnboarding'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase2PlanningReflecting'), \
             patch('elastic_gumby_universal_orch_agent_prototype.agent_main.Phase3TransformExecution'), \
             patch('builtins.print'):
            
            agent = AgentMainInterface()
            
            # Verify session ID format and consistency
            assert agent.session_id.startswith("utoa_")
            assert len(agent.session_id) == 20  # utoa_ + 8 digits + _ + 6 digits
            assert agent.session_id == agent.session_data["session_id"]
            
            # Verify session directory path contains session ID
            assert str(agent.session_dir).endswith(agent.session_id)
