"""
Workflow Planner Core

Core workflow planning engine with iterative_planning as the central method.
The generate_plan and reflect_plan methods are extracted as utility interfaces.
"""

import json
from datetime import datetime
from typing import Any, Dict, List

from colorama import Fore, Style

from elastic_gumby_universal_orch_agent_prototype.data_schema import get_workflow_schema
from elastic_gumby_universal_orch_agent_prototype.visualizer.workflow_loader import WorkflowLoader

from .bedrock_client_manager import BedrockClientManager
from .workflow_processor import WorkflowProcessor


class IterativePlanner:
    """
    Core workflow planning engine that handles the iterative planning process.

    This class focuses on the core iterative_planning method, while
    generate_plan and reflect_plan are provided as utility interfaces.
    """

    def __init__(self, model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", max_interactions=20, max_tokens=8000):
        """
        Initialize the IterativePlanner with required components and configurations.

        Args:
            model_id (str): The Bedrock model ID to use for planning operations.
                          Defaults to Claude 3.7 Sonnet model.
        """
        self.model_id = model_id
        self.max_interactions = max_interactions
        self.max_tokens = max_tokens

        # Initialize Claude messages history for session tracking
        self.claude_messages = {}

        # Initialize specialized components
        self.bedrock_manager = BedrockClientManager()
        self.workflow_processor = WorkflowProcessor()

        # Load schema definitions using centralized schema utilities
        self.workflow_schema = get_workflow_schema()

        self.workflowLoader = None

        print(f"{Fore.GREEN}✓ IterativePlanner initialized successfully{Style.RESET_ALL}")

    def iterative_planning(self, messages, system_prompt, workflow_execution_tool, available_tools):
        """
        Execute the core iterative planning process to generate a complete workflow.

        This method orchestrates the main planning loop, handling model interactions,
        tool usage, and workflow section generation until completion or max iterations.

        Args:
            messages (list): Conversation messages history for the planning session
            system_prompt (str): System prompt defining the planning context and rules
            workflow_execution_tool (dict): Tool definition for workflow execution capabilities
            available_tools (list): list of available tools from Phase 1

        Returns:
            dict: Complete workflow plan with all sections combined, or error dict if failed
        """
        # Initialize planning state
        workflow_sections: List[Dict[str, Any]] = []
        final_metadata = None
        tools_definition = {tool["name"]: tool["parameters"] for tool in available_tools}
        self.workflowLoader = WorkflowLoader(use_colors=True, tools_definition=tools_definition)

        # Execute main planning loop
        interaction_count = 0
        while interaction_count < self.max_interactions:
            interaction_count += 1
            print(
                f"\n{Fore.YELLOW}Interaction {interaction_count}: LLM analyzing and planning...{Style.RESET_ALL}"
            )

            # Get model response
            response = self.bedrock_manager.invoke_model(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                tools=[workflow_execution_tool],
                model_id=self.model_id,
            )

            if "error" in response:
                print(f"{Fore.RED}Error in model invocation: {response['error']}{Style.RESET_ALL}")
                break

            stop_reason = response.get("stop_reason", "")
            content_list = response.get("content", [])

            # Process response content
            if stop_reason == "tool_use":
                self._process_tool_use(content_list, workflow_sections, messages)
            elif stop_reason == "end_turn":
                final_metadata = self._process_final_message(content_list, messages)
                break
            else:
                print(
                    f"{Fore.RED}Unexpected stop reason: {stop_reason}. Continuing with next interaction.{Style.RESET_ALL}"
                )
                pass

        if interaction_count >= self.max_interactions:
            print(f"{Fore.RED}Warning: Maximum interactions ({self.max_interactions}) reached without end_turn completion{Style.RESET_ALL}")

        # Store conversation history and build final workflow
        self.claude_messages = {
            "timestamp": datetime.now().isoformat(),
            "interaction_count": interaction_count,
            "messages": messages,
        }
        return self._build_final_workflow(workflow_sections, final_metadata)

    def _get_self_reflection_message(self, section_number):
        """
        Generate a self-reflection prompt for workflow section validation.

        Creates a detailed validation checklist for the LLM to review and correct
        the generated workflow section before proceeding to the next step.
        
        Legacy workflow validation guidance:
        1. VariableReference/SingleVariableReference VALIDATION:
        - Every variable reference must be a valid JSONata expression wrapped in {{% ... %}} syntax, without any array access [] or function calls ().
        - The "left" property of condition MUST be in the form {{% $variableName %}} or {{% $variableName.property %}}, same for "right" property if it is a variable reference
        - Referenced variableName must have been defined by a previous outputVariable.

        2. TOOL_CALL VALIDATION:
        - Verify every tool_call uses only tools from the AVAILABLE_TOOLS list
        - Parameter values with variable references must be a JSONata string in pattern {{% $variableName %}} or {{% $variableName.property %}}, never concatenate static strings and {{% ... %}} blocks
        - JSONata expressions inside {{% ... %}} must NOT contain any function calls with parentheses () or brackets [], only direct variable references.

        3. STRUCTURE VALIDATION:
        - Remove unnecessary nested containers (sequences with only one step or parallel with only one branch)
        - Remove branches with meaningless condition, e.g. "true == true" or missing "right" property
        - Type of "right" property must be a single variable reference of pattern {{% $variableName %}} or a static number.

        Args:
            section_number (int): The section number to reflect upon

        Returns:
            str: Formatted self-reflection prompt with validation criteria
        """
        return f"""BEFORE CONTINUING, please perform detailed self-reflection on previous generated workflow section {section_number} in "tool_use" content of "assistant" message:
   
DESCRIPTION/FEEDBACK ALIGNMENT:
    - Ensure the generated workflow sections strictly align with the WORKFLOW_DESCRIPTION/USER_FEEDBACK provided in first user message.

After completing this self-reflection:
- If find issues, correct them via the TOOL_USE stop with "section_update" property, i.e. an 1-indexed section number indicating which previous section to update
- Else if have more sections, continue with next section via TOOL_USE stop without "section_update" property
- Else respond with COMPLETION_SIGNAL"""

    def _process_tool_use(self, content_list, workflow_sections, messages):
        """
        Process tool use responses from the LLM during workflow generation.

        Handles workflow section creation or updates when the LLM uses the workflow
        execution tool. Validates the tool input structure and routes to appropriate
        processing methods based on whether it's a new section or section update.

        Args:
            content_list (list): List of content items from the LLM response
            workflow_sections (list): Current list of workflow sections
            messages (list): Conversation messages to update with tool results
        """
        tool_use_id = ""
        assistant_text = ""
        section_number = 0
        workflow_plan = {}

        for content_item in content_list:
            if content_item.get("type") == "text":
                text_content = content_item.get("text", "")
                assistant_text += text_content
            elif content_item.get("type") == "tool_use":
                workflow_plan = content_item.get("input", {})
                tool_use_id = content_item.get("id", "")
                # 1-based section count
                section_number = workflow_plan.get("section_update", len(workflow_sections) + 1)
        
        if assistant_text:
            self._print_assistant_text(assistant_text, f"Workflow Section {section_number} Reasoning Statement")

        # Prepare message structure for tool result
        messages.append({"role": "assistant", "content": content_list})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "To be decided...",
                    }
                ],
            }
        )

        # Load and validate workflow plan
        result = self.workflowLoader.load_workflow_from_json_string(json.dumps(workflow_plan))
        if result["success"] is False:
            errors_str = '\n'.join(result['errors'])
            messages[-1]["content"][0]["content"] = f"Invalid tool_input for workflow section {section_number}. Check the following errors:\n{errors_str}"
            return
        workflow_plan = result["workflow"] # use the normalized workflow structure
        
        # Route to appropriate processing method
        if section_number <= len(workflow_sections):
            self._process_section_update(
                section_number, workflow_sections, workflow_plan
            )
        else:
            self._process_new_section(
                section_number, workflow_sections, workflow_plan
            )

        # Set success response with self-reflection prompt
        messages[-1]["content"][0]["content"] = (
            f"Workflow section {section_number} received and recorded.\n"
            f"{self._get_self_reflection_message(section_number)}"
        )

    def _process_final_message(self, content_list, messages):
        """
        Process the final completion message from the LLM.

        Handles the end-of-planning response, extracting any final metadata
        such as workflow name and description from the completion message.

        Args:
            content_list (list): List of content items from the final LLM response
            messages (list): Conversation messages to append the final response

        Returns:
            dict: Extracted final metadata (name, description, etc.) or empty dict
        """
        final_metadata = {}
        text_parts = [item["text"] for item in content_list if item.get("type") == "text"]
        full_text = "".join(text_parts).strip()

        if full_text:
            self._print_assistant_text(full_text, "Final Assistant Message")
            final_metadata = self.workflow_processor.extract_final_metadata(full_text)
            if final_metadata:
                print(
                    f"{Fore.GREEN}✓ Extracted final metadata: {final_metadata['name']}{Style.RESET_ALL}"
                )

        messages.append({"role": "assistant", "content": content_list})
        return final_metadata

    def _process_section_update(
        self, section_number, workflow_sections, workflow_plan
    ):
        """
        Process an update to an existing workflow section.

        Updates a previously generated workflow section with corrected or refined
        workflow plan data, typically after self-reflection validation.

        Args:
            section_number (int): The 1-based section number to update
            workflow_sections (list): List of workflow sections to modify
            workflow_plan (dict): New workflow plan data to replace the existing section
        """

        # Update the existing section
        section_index = section_number - 1
        workflow_sections[section_index]["workflow_plan"] = workflow_plan
        print(f"{Fore.GREEN}✓ Updated workflow section {section_number}{Style.RESET_ALL}\n")

    def _process_new_section(
        self, section_number, workflow_sections, workflow_plan
    ):
        """
        Process the creation of a new workflow section.

        Adds a new workflow section to the sections list with the provided
        workflow plan data and section metadata.

        Args:
            section_number (int): The 1-based section number being created
            workflow_sections (list): List of workflow sections to append to
            workflow_plan (dict): Workflow plan data for the new section
        """
        # Add new section
        workflow_sections.append({"section_number": section_number, "workflow_plan": workflow_plan})
        print(f"{Fore.GREEN}✓ Generated workflow section {section_number}{Style.RESET_ALL}\n")

    def _build_final_workflow(self, workflow_sections, final_metadata):
        """
        Build the complete final workflow from all generated sections.

        Combines all workflow sections into a single cohesive workflow and applies
        any final metadata (name, description) extracted from the completion message.

        Args:
            workflow_sections (list): List of all generated workflow sections
            final_metadata (dict): Final metadata to apply to the complete workflow

        Returns:
            dict: Complete workflow plan with all sections combined and metadata applied,
                  or error dict if no valid sections were generated
        """
        if not workflow_sections:
            print(f"{Fore.RED}Error: No valid workflow sections generated{Style.RESET_ALL}")
            return {"error": "No valid workflow sections generated"}

        # Combine workflow sections
        print(
            f"\n{Fore.BLUE}Planning completed: {len(workflow_sections)} section(s) workflow generated{Style.RESET_ALL}"
        )
        complete_workflow = self.workflow_processor.combine_workflow_sections(workflow_sections)

        # Update name and description if provided
        if final_metadata:
            complete_workflow.update(final_metadata)

        return complete_workflow

    def _print_assistant_text(self, assistant_text, context=""):
        """
        Display LLM assistant text with proper formatting and visual separation.

        Provides a consistent, readable format for displaying LLM responses with
        optional context headers and visual separators for better readability.

        Args:
            assistant_text (str): The text content from the LLM assistant to display
            context (str, optional): Context description to show as a header.
                                    Defaults to "LLM Response" if not provided.
        """
        print(f"{Fore.CYAN}{context}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}")
        print(assistant_text.strip())
        print(f"{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}")
