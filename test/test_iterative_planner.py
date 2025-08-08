"""
Tests for IterativePlanner

Tests designed for Coverlay compatibility:
- BrazilPython-Pytest-6.x
- Python-Pytest-cov-3.x and Coverage-6.x
- BrazilPythonTestSupport-3.0
"""

from unittest.mock import Mock, patch

from elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner import IterativePlanner


class TestGetSelfReflectionMessage:
    """Test _get_self_reflection_message method."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.BedrockClientManager"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.WorkflowProcessor"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.get_workflow_schema"
        ), patch(
            "builtins.print"
        ):
            self.planner = IterativePlanner()

    def test_get_self_reflection_message_format(self):
        """Test self-reflection message format and content."""
        section_number = 1

        result = self.planner._get_self_reflection_message(section_number)

        assert "BEFORE CONTINUING" in result
        assert "section 1" in result
        assert "DESCRIPTION/FEEDBACK ALIGNMENT" in result
        assert "section_update" in result
        assert "COMPLETION_SIGNAL" in result

    def test_get_self_reflection_message_different_sections(self):
        """Test self-reflection message with different section numbers."""
        result_1 = self.planner._get_self_reflection_message(1)
        result_5 = self.planner._get_self_reflection_message(5)

        assert "section 1" in result_1
        assert "section 5" in result_5
        assert result_1 != result_5


class TestBuildFinalWorkflow:
    """Test _build_final_workflow method."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.BedrockClientManager"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.WorkflowProcessor"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.get_workflow_schema"
        ), patch(
            "builtins.print"
        ):
            self.planner = IterativePlanner()

    @patch("builtins.print")
    def test_build_final_workflow_empty_sections(self, mock_print):
        """Test building final workflow with empty sections."""
        workflow_sections = []
        final_metadata = {}

        result = self.planner._build_final_workflow(workflow_sections, final_metadata)

        assert result == {"error": "No valid workflow sections generated"}
        mock_print.assert_called()

    @patch("builtins.print")
    def test_build_final_workflow_with_sections(self, mock_print):
        """Test building final workflow with valid sections."""
        workflow_sections = [
            {"section_number": 1, "workflow_plan": {"root": {"type": "tool_call"}}}
        ]
        final_metadata = {"name": "Test Workflow", "description": "Test Description"}

        # Mock the workflow processor
        mock_combined = {"root": {"type": "tool_call"}}
        self.planner.workflow_processor.combine_workflow_sections = Mock(return_value=mock_combined)

        result = self.planner._build_final_workflow(workflow_sections, final_metadata)

        expected_result = {
            "name": "Test Workflow",
            "description": "Test Description",
            "root": {"type": "tool_call"},
        }
        assert result == expected_result
        self.planner.workflow_processor.combine_workflow_sections.assert_called_once_with(
            workflow_sections
        )
        mock_print.assert_called()


class TestProcessToolUse:
    """Test _process_tool_use method."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.BedrockClientManager"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.WorkflowProcessor"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.get_workflow_schema"
        ), patch(
            "builtins.print"
        ):
            self.planner = IterativePlanner()

    @patch.object(IterativePlanner, "_process_new_section")
    @patch.object(IterativePlanner, "_get_self_reflection_message")
    @patch.object(IterativePlanner, "_print_assistant_text")
    def test_process_tool_use_new_section(self, mock_print_assistant, mock_reflection, mock_process_new):
        """Test processing tool use for new section creation."""
        mock_reflection.return_value = "reflection message"

        content_list = [
            {"type": "text", "text": "Creating new section"},
            {
                "type": "tool_use",
                "id": "tool_123",
                "input": {"root": {"type": "tool_call", "name": "test_tool"}},
            },
        ]
        workflow_sections = []
        messages = []

        # Mock WorkflowLoader to return success
        mock_workflow_loader = Mock()
        mock_workflow_loader.load_workflow_from_json_string.return_value = {
            "success": True,
            "workflow": {"root": {"type": "tool_call", "name": "test_tool"}}
        }
        self.planner.workflowLoader = mock_workflow_loader

        self.planner._process_tool_use(content_list, workflow_sections, messages)

        # Check messages were updated correctly
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"][0]["tool_use_id"] == "tool_123"
        assert "Workflow section 1 received and recorded" in messages[1]["content"][0]["content"]

        # Check that new section processing was called
        mock_process_new.assert_called_once()
        mock_reflection.assert_called_once_with(1)
        mock_print_assistant.assert_called_once()

    @patch.object(IterativePlanner, "_process_section_update")
    @patch.object(IterativePlanner, "_get_self_reflection_message")
    @patch.object(IterativePlanner, "_print_assistant_text")
    def test_process_tool_use_section_update(self, mock_print_assistant, mock_reflection, mock_process_update):
        """Test processing tool use for section update."""
        mock_reflection.return_value = "reflection message"

        content_list = [
            {"type": "text", "text": "Updating section"},
            {
                "type": "tool_use",
                "id": "tool_456",
                "input": {
                    "section_update": 1,
                    "root": {"type": "tool_call", "name": "updated_tool"},
                },
            },
        ]
        workflow_sections = [{"section_number": 1, "workflow_plan": {"old": "plan"}}]
        messages = []

        # Mock WorkflowLoader to return success
        mock_workflow_loader = Mock()
        mock_workflow_loader.load_workflow_from_json_string.return_value = {
            "success": True,
            "workflow": {"root": {"type": "tool_call", "name": "updated_tool"}}
        }
        self.planner.workflowLoader = mock_workflow_loader

        self.planner._process_tool_use(content_list, workflow_sections, messages)

        # Check that section update processing was called
        mock_process_update.assert_called_once()
        mock_reflection.assert_called_once_with(1)

    def test_process_tool_use_workflow_loader_validation_failure(self):
        """Test processing tool use with WorkflowLoader validation failure."""
        content_list = [
            {
                "type": "tool_use",
                "id": "tool_789",
                "input": {"name": "test", "description": "test"},  # Invalid workflow
            }
        ]
        workflow_sections = []
        messages = []

        # Mock WorkflowLoader to return failure
        mock_workflow_loader = Mock()
        mock_workflow_loader.load_workflow_from_json_string.return_value = {
            "success": False,
            "errors": ["Missing root property", "Invalid structure"]
        }
        self.planner.workflowLoader = mock_workflow_loader

        self.planner._process_tool_use(content_list, workflow_sections, messages)

        # Check error message was set
        assert len(messages) == 2
        assert "Invalid tool_input for workflow section 1" in messages[1]["content"][0]["content"]
        assert "Missing root property" in messages[1]["content"][0]["content"]
        assert "Invalid structure" in messages[1]["content"][0]["content"]

    @patch.object(IterativePlanner, "_process_new_section")
    @patch.object(IterativePlanner, "_get_self_reflection_message")
    def test_process_tool_use_text_extraction(self, mock_reflection, mock_process_new):
        """Test processing tool use with text content extraction."""
        mock_reflection.return_value = "reflection message"

        content_list = [
            {"type": "text", "text": "First part "},
            {"type": "text", "text": "Second part"},
            {
                "type": "tool_use",
                "id": "tool_text",
                "input": {"root": {"type": "tool_call", "name": "text_tool"}},
            },
        ]
        workflow_sections = []
        messages = []

        # Mock WorkflowLoader to return success
        mock_workflow_loader = Mock()
        mock_workflow_loader.load_workflow_from_json_string.return_value = {
            "success": True,
            "workflow": {"root": {"type": "tool_call", "name": "text_tool"}}
        }
        self.planner.workflowLoader = mock_workflow_loader

        with patch.object(self.planner, "_print_assistant_text") as mock_print_assistant:
            self.planner._process_tool_use(content_list, workflow_sections, messages)

            # Check that text was concatenated and printed
            mock_print_assistant.assert_called_once_with(
                "First part Second part", "Workflow Section 1 Reasoning Statement"
            )


class TestProcessFinalMessage:
    """Test _process_final_message method."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.BedrockClientManager"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.WorkflowProcessor"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.get_workflow_schema"
        ), patch(
            "builtins.print"
        ):
            self.planner = IterativePlanner()

    @patch.object(IterativePlanner, "_print_assistant_text")
    @patch("builtins.print")
    def test_process_final_message_with_metadata(self, mock_print, mock_print_assistant):
        """Test processing final message with extractable metadata."""
        content_list = [{"type": "text", "text": "Final completion message with metadata"}]
        messages = []

        # Mock the workflow processor to return metadata
        mock_metadata = {"name": "Final Workflow", "description": "Final Description"}
        self.planner.workflow_processor.extract_final_metadata = Mock(return_value=mock_metadata)

        result = self.planner._process_final_message(content_list, messages)

        assert result == mock_metadata
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        mock_print_assistant.assert_called_once()
        mock_print.assert_called()

    @patch.object(IterativePlanner, "_print_assistant_text")
    def test_process_final_message_without_metadata(self, mock_print_assistant):
        """Test processing final message without extractable metadata."""
        content_list = [{"type": "text", "text": "Final completion message"}]
        messages = []

        # Mock the workflow processor to return empty metadata
        self.planner.workflow_processor.extract_final_metadata = Mock(return_value={})

        result = self.planner._process_final_message(content_list, messages)

        assert result == {}
        mock_print_assistant.assert_called_once_with(
            "Final completion message", "Final Assistant Message"
        )

    def test_process_final_message_multiple_text_items(self):
        """Test processing final message with multiple text content items."""
        content_list = [
            {"type": "text", "text": "First part "},
            {"type": "text", "text": "Second part"},
            {"type": "other", "data": "ignored"},
        ]
        messages = []

        self.planner.workflow_processor.extract_final_metadata = Mock(return_value={})

        with patch.object(self.planner, "_print_assistant_text") as mock_print_assistant:
            result = self.planner._process_final_message(content_list, messages)

            # Should concatenate text parts
            mock_print_assistant.assert_called_once_with(
                "First part Second part", "Final Assistant Message"
            )


class TestIterativePlanning:
    """Test iterative_planning method - the core orchestration method."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.BedrockClientManager"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.WorkflowProcessor"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.get_workflow_schema"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.WorkflowLoader"
        ), patch(
            "builtins.print"
        ):
            self.planner = IterativePlanner()

    @patch("builtins.print")
    def test_iterative_planning_model_error(self, mock_print):
        """Test iterative planning when model returns error."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
        system_prompt = "test prompt"
        workflow_tool = {"name": "workflow_execution"}
        available_tools = [{"name": "test_tool", "parameters": {}}]

        # Mock bedrock manager to return error
        self.planner.bedrock_manager.invoke_model = Mock(return_value={"error": "Model error"})

        result = self.planner.iterative_planning(messages, system_prompt, workflow_tool, available_tools)

        # Should return error result from _build_final_workflow with empty sections
        assert "error" in result
        mock_print.assert_called()

    @patch("builtins.print")
    def test_iterative_planning_end_turn_completion(self, mock_print):
        """Test iterative planning with immediate end_turn completion."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
        system_prompt = "test prompt"
        workflow_tool = {"name": "workflow_execution"}
        available_tools = [{"name": "test_tool", "parameters": {}}]

        # Mock bedrock manager to return end_turn
        mock_response = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Completion message"}],
        }
        self.planner.bedrock_manager.invoke_model = Mock(return_value=mock_response)

        # Mock the process methods
        self.planner.workflow_processor.extract_final_metadata = Mock(return_value={"name": "Test"})
        self.planner.workflow_processor.combine_workflow_sections = Mock(
            return_value={"error": "No valid workflow sections generated"}
        )

        with patch.object(
            self.planner, "_process_final_message", return_value={"name": "Test"}
        ) as mock_final:
            result = self.planner.iterative_planning(messages, system_prompt, workflow_tool, available_tools)

            mock_final.assert_called_once()
            assert "error" in result  # No sections were created

    @patch("builtins.print")
    def test_iterative_planning_tool_use_then_completion(self, mock_print):
        """Test iterative planning with tool use followed by completion."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
        system_prompt = "test prompt"
        workflow_tool = {"name": "workflow_execution"}
        available_tools = [{"name": "test_tool", "parameters": {}}]

        # Mock bedrock manager to return tool_use first, then end_turn
        tool_response = {
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Creating workflow"},
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "input": {"root": {"type": "tool_call", "name": "test_tool"}},
                },
            ],
        }
        end_response = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Workflow complete"}],
        }

        self.planner.bedrock_manager.invoke_model = Mock(side_effect=[tool_response, end_response])

        # Mock the workflow processor
        mock_final_workflow = {"name": "Test Workflow", "root": {"type": "tool_call"}}
        self.planner.workflow_processor.combine_workflow_sections = Mock(
            return_value=mock_final_workflow
        )
        self.planner.workflow_processor.extract_final_metadata = Mock(
            return_value={"description": "Final"}
        )

        # Mock _process_tool_use to simulate adding workflow sections
        def mock_process_tool_use_side_effect(content_list, workflow_sections, messages):
            workflow_sections.append({"section": 1, "content": "test"})

        with patch.object(
            self.planner, "_process_tool_use", side_effect=mock_process_tool_use_side_effect
        ) as mock_tool_use, patch.object(
            self.planner, "_process_final_message", return_value={"description": "Final"}
        ) as mock_final:

            result = self.planner.iterative_planning(messages, system_prompt, workflow_tool, available_tools)

            mock_tool_use.assert_called_once()
            mock_final.assert_called_once()
            assert result["name"] == "Test Workflow"
            assert result["description"] == "Final"

    @patch("builtins.print")
    def test_iterative_planning_bedrock_manager_called_correctly(self, mock_print):
        """Test that bedrock manager is called with correct parameters."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
        system_prompt = "test system prompt"
        workflow_tool = {"name": "workflow_execution", "description": "test tool"}
        available_tools = [{"name": "test_tool", "parameters": {"param1": "value1"}}]

        mock_response = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Done"}]}
        self.planner.bedrock_manager.invoke_model = Mock(return_value=mock_response)

        # Mock workflow processor
        self.planner.workflow_processor.extract_final_metadata = Mock(return_value={})
        self.planner.workflow_processor.combine_workflow_sections = Mock(
            return_value={"error": "No sections"}
        )

        with patch.object(self.planner, "_process_final_message", return_value={}):
            self.planner.iterative_planning(messages, system_prompt, workflow_tool, available_tools)

            # Verify bedrock manager was called with correct parameters
            self.planner.bedrock_manager.invoke_model.assert_called_with(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=8000,
                tools=[workflow_tool],
                model_id=self.planner.model_id,
            )

    @patch("builtins.print")
    def test_iterative_planning_max_interactions_reached(self, mock_print):
        """Test iterative planning when max interactions limit is reached."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]
        system_prompt = "test prompt"
        workflow_tool = {"name": "workflow_execution"}
        available_tools = [{"name": "test_tool", "parameters": {}}]

        # Mock bedrock manager to always return tool_use (never end_turn)
        mock_response = {
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Creating workflow"},
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "input": {"root": {"type": "tool_call", "name": "test_tool"}},
                },
            ],
        }
        self.planner.bedrock_manager.invoke_model = Mock(return_value=mock_response)

        # Mock WorkflowLoader to return success
        mock_workflow_loader = Mock()
        mock_workflow_loader.load_workflow_from_json_string.return_value = {
            "success": True,
            "workflow": {"root": {"type": "tool_call", "name": "test_tool"}}
        }
        self.planner.workflowLoader = mock_workflow_loader

        # Mock workflow processor
        self.planner.workflow_processor.combine_workflow_sections = Mock(
            return_value={"name": "Max Interactions Workflow"}
        )

        with patch.object(self.planner, "_process_tool_use") as mock_tool_use:
            result = self.planner.iterative_planning(messages, system_prompt, workflow_tool, available_tools)

            # Should have called bedrock manager 20 times (max_interactions)
            assert self.planner.bedrock_manager.invoke_model.call_count == 20
            
            # Should print warning about max interactions
            warning_calls = [call for call in mock_print.call_args_list 
                           if "Maximum interactions" in str(call)]
            assert len(warning_calls) > 0

class TestProcessNewSection:
    """Test _process_new_section method."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.BedrockClientManager"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.WorkflowProcessor"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.get_workflow_schema"
        ), patch(
            "builtins.print"
        ):
            self.planner = IterativePlanner()

    @patch("builtins.print")
    def test_process_new_section(self, mock_print):
        """Test processing a new workflow section."""
        section_number = 1
        workflow_sections = []
        workflow_plan = {"root": {"type": "tool_call", "name": "test_tool"}}

        self.planner._process_new_section(section_number, workflow_sections, workflow_plan)

        # Check that section was added
        assert len(workflow_sections) == 1
        assert workflow_sections[0]["section_number"] == 1
        assert workflow_sections[0]["workflow_plan"] == workflow_plan

        # Check that success message was printed
        success_calls = [call for call in mock_print.call_args_list 
                        if "Generated workflow section 1" in str(call)]
        assert len(success_calls) > 0

    @patch("builtins.print")
    def test_process_new_section_multiple_sections(self, mock_print):
        """Test processing multiple new workflow sections."""
        workflow_sections = []
        
        # Add first section
        workflow_plan_1 = {"root": {"type": "tool_call", "name": "tool_1"}}
        self.planner._process_new_section(1, workflow_sections, workflow_plan_1)
        
        # Add second section
        workflow_plan_2 = {"root": {"type": "tool_call", "name": "tool_2"}}
        self.planner._process_new_section(2, workflow_sections, workflow_plan_2)

        # Check that both sections were added
        assert len(workflow_sections) == 2
        assert workflow_sections[0]["section_number"] == 1
        assert workflow_sections[0]["workflow_plan"] == workflow_plan_1
        assert workflow_sections[1]["section_number"] == 2
        assert workflow_sections[1]["workflow_plan"] == workflow_plan_2


class TestProcessSectionUpdate:
    """Test _process_section_update method."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.BedrockClientManager"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.WorkflowProcessor"
        ), patch(
            "elastic_gumby_universal_orch_agent_prototype.planner.iterative_planner.get_workflow_schema"
        ), patch(
            "builtins.print"
        ):
            self.planner = IterativePlanner()

    @patch("builtins.print")
    def test_process_section_update(self, mock_print):
        """Test updating an existing workflow section."""
        section_number = 1
        workflow_sections = [
            {"section_number": 1, "workflow_plan": {"root": {"type": "old_tool"}}}
        ]
        new_workflow_plan = {"root": {"type": "tool_call", "name": "updated_tool"}}

        self.planner._process_section_update(section_number, workflow_sections, new_workflow_plan)

        # Check that section was updated
        assert len(workflow_sections) == 1
        assert workflow_sections[0]["section_number"] == 1
        assert workflow_sections[0]["workflow_plan"] == new_workflow_plan

        # Check that update message was printed
        update_calls = [call for call in mock_print.call_args_list 
                       if "Updated workflow section 1" in str(call)]
        assert len(update_calls) > 0

    @patch("builtins.print")
    def test_process_section_update_middle_section(self, mock_print):
        """Test updating a middle section in a multi-section workflow."""
        workflow_sections = [
            {"section_number": 1, "workflow_plan": {"root": {"type": "tool_1"}}},
            {"section_number": 2, "workflow_plan": {"root": {"type": "tool_2"}}},
            {"section_number": 3, "workflow_plan": {"root": {"type": "tool_3"}}}
        ]
        new_workflow_plan = {"root": {"type": "updated_tool_2"}}

        self.planner._process_section_update(2, workflow_sections, new_workflow_plan)

        # Check that only the middle section was updated
        assert len(workflow_sections) == 3
        assert workflow_sections[0]["workflow_plan"]["root"]["type"] == "tool_1"  # unchanged
        assert workflow_sections[1]["workflow_plan"] == new_workflow_plan  # updated
        assert workflow_sections[2]["workflow_plan"]["root"]["type"] == "tool_3"  # unchanged
