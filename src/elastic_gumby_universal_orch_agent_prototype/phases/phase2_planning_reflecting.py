"""
Phase 2: Planning & User Reflection - Updated with Visualizer Module

This module handles workflow planning based on user descriptions
and iterative refinement through user feedback.

Updated to use the new visualizer module instead of the old data_test/workflow_visualizer.
"""

import json
import copy
from typing import Any, Dict, Optional

from colorama import Fore, Style

from elastic_gumby_universal_orch_agent_prototype.planner.utils import generate_plan, reflect_plan


class Phase2PlanningReflecting:
    """
    Handles Phase 2: Planning & User Reflection

    Creates workflow plans based on user descriptions and iterates
    on them based on user feedback until approval.

    Updated to use the new visualizer module for enhanced workflow visualization.
    """

    def __init__(self, session_data: Dict[str, Any], get_user_input_func, workflow_visualizer):
        """
        Initialize Phase 2 handler.

        Args:
            session_data: Reference to main session data dictionary
            get_user_input_func: Function to get user input
        """
        self.session_data = session_data
        self.get_user_input = get_user_input_func
        self.workflow_visualizer = workflow_visualizer
        self.model_id = None
        self.max_interactions = None
        self.max_tokens = None

    def run(self) -> Optional[str]:
        """
        Execute Phase 2: Planning & User Reflection.

        Returns:
            str: 'next' to proceed to Phase 3, 'back' to return to Phase 1, None to exit
        """
        self.print_phase_header()

        # Check if we have tools from Phase 1
        available_tools = self.session_data.get("tools", [])
        if not available_tools:
            print(f"\n{Fore.RED}❌ No tools available from Phase 1!")
            print(f"You need to complete Phase 1 (Tools Onboarding) first.{Style.RESET_ALL}")
            return "back"

        print(f"\n{Fore.GREEN}✅ Found {len(available_tools)} available tools from Phase 1{Style.RESET_ALL}")

        # Multiple workflow descriptions
        while True:
            # Step 1: Collect workflow description
            self.print_workflow_guidance()
            workflow_description = self.collect_workflow_description()
            if workflow_description is None:
                return None  # User wants to quit

            # Step 2: Generate initial workflow plan
            self.config_planner()
            generation_success = self.generate_workflow_plan(workflow_description)

            # Feedback iteration for the same workflow description
            while generation_success:
                # Step 3: Display current plan using new visualizer module
                print(f"\n{Fore.CYAN}🤔 Step 3: Workflow Plan Review{Style.RESET_ALL}")
                workflow_plan = self.session_data.get("workflow_plan", {})
                print(self.workflow_visualizer.visualize_workflow(workflow_plan))

                # Step 4: Collect user feedback
                feedback = self.collect_user_feedback()

                # Step 5: Process feedback and determine next action
                action = self.process_feedback(feedback)
                if action in ["next", "back"]:
                    return action
                elif action == "restart":
                    break  # start over with new workflow description
                else: # 'iterate'
                    continue

    def print_phase_header(self):
        """Print the Phase 2 header and introduction."""
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}🤖 PHASE 2: PLANNING & USER REFLECTION")
        print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
        print(f"\n{Fore.WHITE}In this phase, we'll create a detailed Workflow (State Machine) based on your description,")
        print(f"and iterate on it based on your feedback.{Style.RESET_ALL}")

    def print_workflow_guidance(self):
        """Print guidance for workflow description."""
        print(f"\n{Fore.CYAN}📝 Step 1: Workflow Description Input{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Please provide a detailed description of the workflow you want to orchestrate.")
        print(f"Include information about:")
        print(f"• The overall starting and ending points of the workflow")
        print(f"• Flow of tasks and their relationships")
        print(f"• Inputs and outputs dependency between different tasks")
        print(f"• Error handling requirements{Style.RESET_ALL}")

    def collect_workflow_description(self) -> Optional[str]:
        """
        Collect workflow description from user.

        Returns:
            str: Workflow description, or None if user wants to quit
        """
        user_input = self.get_user_input("Describe the topology of your desired workflow:", "file")
        if user_input.lower() == "quit":
            return None

        # Check if input is JSON format
        workflow_description = self._extract_workflow_description_from_input(user_input)
        self.session_data["workflow_description"] = workflow_description
        return workflow_description

    def _extract_workflow_description_from_input(self, user_input: str) -> str:
        """
        Extract workflow description from user input, handling both plain text and JSON formats.

        Args:
            user_input (str): Raw user input

        Returns:
            str: Extracted workflow description
        """
        try:
            parsed_input = json.loads(user_input.strip())
            # If it's a dictionary, look for common workflow description keys
            if isinstance(parsed_input, dict):
                for key in ["workflow_description", "description"]:
                    if key in parsed_input:
                        return str(parsed_input[key])
                # If no specific key found, convert the entire dict to a descriptive string
                return f"{json.dumps(parsed_input, indent=2)}"
            # convert to string
            else: 
                return str(parsed_input)

        except json.JSONDecodeError:
            # Not JSON, treat as plain text workflow description
            return user_input.strip()
        
    def config_planner(self):
        self.model_id = self.get_user_input("Enter the model ID for the planner: (Default 'us.anthropic.claude-3-7-sonnet-20250219-v1:0')", "text")
        if not self.model_id:
            self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        self.max_interactions = self.get_user_input("Enter the maximum LLM interactions limit: (Default 20)", "text")
        if not self.max_interactions:
            self.max_interactions = 20
        self.max_tokens = self.get_user_input("Enter the maximum tokens limit per response: (Default 8000)", "text")
        if not self.max_tokens:
            self.max_tokens = 8000

    def generate_workflow_plan(self, description: str) -> bool:
        """
        Generate workflow plan using Workflow Planner Package or placeholder.

        Args:
            description: Workflow description from user

        Returns:
            Dict containing the workflow plan
        """
        print(f"\n{Fore.CYAN}🧠 Step 2: Generating Workflow Plan{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🤖 Analyzing your workflow description and generating a structured plan...{Style.RESET_ALL}")

        # Get available tools and remove 'resource' key if present
        available_tools = copy.deepcopy(self.session_data.get("tools", []))
        for tool in available_tools:
            tool.pop("resource", None)

        try:
            print(f"{Fore.CYAN}Using Workflow Planner with {len(available_tools)} available tools...{Style.RESET_ALL}")

            # Generate the workflow plan
            workflow_plan, claude_messages = generate_plan(
                workflow_description=description, 
                available_tools=available_tools, 
                model_id=self.model_id,
                max_interactions=self.max_interactions, 
                max_tokens=self.max_tokens
            )
            self.session_data["workflow_plan"] = workflow_plan
            self.session_data["claude_messages"].append(claude_messages) 
            return True
        
        except Exception as e:
            print(f"{Fore.RED}❌ Error generating workflow plan with Planner: {e}{Style.RESET_ALL}")
            return False
    
    def collect_user_feedback(self) -> str:
        """
        Collect feedback from user about the workflow plan.

        Returns:
            User feedback string
        """
        print(f"\n{Fore.YELLOW}Please review the generated workflow plan and provide feedback:{Style.RESET_ALL}")
        print(f"You can provide detailed multi-line feedback about above workflow.")
        print(f"Special commands:")
        print(f" • 'approve' - Accept the plan and continue to Phase 3")
        print(f" • 'back' - Return to Phase 1 for more tools")
        print(f" • 'restart' - Start over with a new workflow description")

        feedback = self.get_user_input("Your feedback:", "text")
        return feedback


    def process_feedback(self, feedback: str) -> str:
        """
        Process user feedback and determine next action.

        Args:
            feedback: User feedback string

        Returns:
            Action to take: 'next', 'back', 'restart', or 'iterate'
        """
        feedback_lower = feedback.lower().strip()

        if feedback_lower == "approve":
            print(f"{Fore.GREEN}✅ Workflow plan approved. Ready for Phase 3: Execution & Monitor{Style.RESET_ALL}")
            return "next"
        elif feedback_lower == "back":
            print(f"{Fore.YELLOW}↩️ Returning to Phase 1 for additional tool information{Style.RESET_ALL}")
            return "back"
        elif feedback_lower == "restart":
            print(f"{Fore.CYAN}🔄  Starting over with a new workflow description...{Style.RESET_ALL}")
            return "restart"
        else:
            self.reflect_workflow_plan(feedback)
            return "iterate"

    def reflect_workflow_plan(self, feedback: str) -> bool:
        """
        Handle iterative feedback by reflecting on and updating the existing plan.

        Args:
            feedback: User feedback to incorporate

        Returns:
            Dict[str, Any]: Updated workflow plan dictionary, or None if update failed
        """
        print(f"\n{Fore.YELLOW}Incorporating feedback and updating plan...{Style.RESET_ALL}")

        # Get the current workflow plan
        current_plan = self.session_data.get("workflow_plan", {})
        available_tools = copy.deepcopy(self.session_data.get("tools", []))
        for tool in available_tools:
            tool.pop("resource", None)

        try:
            print(f"{Fore.BLUE}🤖 Using Workflow Planner Package to reflect on and update the plan...{Style.RESET_ALL}")
            # Use the reflect_plan method to update the workflow based on feedback
            updated_plan, claude_messages = reflect_plan(
                existing_workflow_plan=current_plan,
                user_feedback=feedback,
                available_tools=available_tools,
                model_id=self.model_id,
                max_interactions=self.max_interactions,
                max_tokens=self.max_tokens
            )
            self.session_data["workflow_plan"] = updated_plan
            self.session_data["claude_messages"].append(claude_messages) 
            return True
        except Exception as e:
            print(f"{Fore.RED}❌ Exception during plan reflection: {str(e)}\nTry providing different feedback.{Style.RESET_ALL}")
            return False