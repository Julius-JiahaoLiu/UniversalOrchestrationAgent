"""
Phase 3: Execution & Monitor

This module handles tool binding, code translation, backup,
and workflow execution with real-time monitoring.
"""

import json
import boto3
import random
import re
from time import sleep
from pathlib import Path
from typing import Any, Dict, Optional
from elastic_gumby_universal_orch_agent_prototype.transform.state_machine_transformer import StateMachineTransformer

from colorama import Fore, Style


class Phase3TransformExecution:
    """
    Handles Phase 3: Execution & Monitor

    Manages tool binding, code translation, backup operations,
    and workflow execution with progress tracking.
    """

    def __init__(self, session_data: Dict[str, Any], get_user_input_func, session_dir: Path):
        """
        Initialize Phase 3 handler.

        Args:
            session_data: Reference to main session data dictionary
            get_user_input_func: Function to get user input
            session_dir: Path to session directory for backups
        """
        self.session_data = session_data
        self.get_user_input = get_user_input_func
        self.session_dir = session_dir
        self.SFN_client = None

    def print_phase_header(self):
        """Print the Phase 3 header and introduction."""
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}⚙️ PHASE 3: Transform & EXECUTION")
        print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
        print(f"\n{Fore.WHITE}In this phase, we'll transform your workflow plan to state machine,")
        print(f"and execute with real-time monitoring.{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}🎭 Step 1: Workflow Transformation{Style.RESET_ALL}")

    def run(self) -> Optional[str]:
        """
        Execute Phase 3: Execution & Monitor.

        Returns:
            str: Next action ('back', 'restart', None for exit)
        """
        self.print_phase_header()

        # Step 1: Transform Workflow
        choice = self.get_user_input("Would you like to transform workflow into State Machine? (y/n):", "text")
        if choice.lower() == 'y':
            transformer = StateMachineTransformer(self.session_data.get("tools", []))
            transformer.save_state_machine(self.session_data.get("workflow_plan", {}), self.session_dir)
        else:
            return self.handle_post_execution_options()

        # Step 2: Deploy & Execution
        state_machine_arn = self.deploy_state_machine()
        if state_machine_arn:
            self.execute_state_machine(state_machine_arn)
            self.delete_state_machine(state_machine_arn)

        # Handle post-execution options
        return self.handle_post_execution_options()

    def deploy_state_machine(self) -> str:
        """
        Deploy the state machine to AWS Step Functions.
        
        Returns:
            str: ARN of the deployed state machine, or empty string if skipped
        """
        print(f"\n{Fore.CYAN}🚀 Step 2: Workflow Execution{Style.RESET_ALL}")
        duplicate_name_counter = 0
        while True:
            deploy_choice = self.get_user_input("Would you like to deploy this state machine to AWS Step Functions? (y/n):", "text")
            if deploy_choice == 'y':
                if not self.SFN_client:
                    self.SFN_client = boto3.client('stepfunctions')
                role_arn = self.get_user_input("Enter the IAM role ARN for the state machine: ", "text")
                with open(self.session_dir / "state_machine.asl.json", 'r') as file:
                    definition = json.load(file)
                duplicate_name_counter += 1
                state_machine_name = re.sub(r"\s+", "_", self.session_data.get("workflow_plan", {}).get("name", "Test State Machine")) # Replace spaces with underscores
                state_machine_name += f"_v{duplicate_name_counter}"  # Append version number to avoid duplicates
                try:
                    response = self.SFN_client.create_state_machine(
                        name=state_machine_name,
                        definition=json.dumps(definition),
                        roleArn=role_arn,
                        type="STANDARD"
                    )
                    state_machine_arn = response.get("stateMachineArn")
                    print(f"{Fore.GREEN}State Machine ARN: {state_machine_arn} deployed successfully!{Style.RESET_ALL}")
                    return state_machine_arn
                except Exception as e:
                    print(f"{Fore.RED}❌ Error deploying state machine: {e}{Style.RESET_ALL}")
                    continue  # Retry deployment
            else:
                print(f"{Fore.YELLOW}Skipping state machine deployment.{Style.RESET_ALL}")
                return ""
            
    def execute_state_machine(self, state_machine_arn: str):
        """
        Execute the state machine with real-time progress tracking.

        Args:
            state_machine_arn: ARN of the deployed state machine

        Returns:
            bool: True if execution successful, False otherwise
        """
        while True:
            execution_choice = self.get_user_input("Would you like to start a new execution with simulated input? (y/n):", "text")
            if execution_choice == 'y':
                with open(self.session_dir / "exec_input.json", 'r') as file:
                    execution_input = json.load(file)
                execution_input = self._random_choose_execution_input(execution_input)
                print(f"Simulated execution input: {execution_input}")
                try:
                    response = self.SFN_client.start_execution(
                        stateMachineArn=state_machine_arn,
                        input=json.dumps(execution_input)
                    )
                    execution_arn = response.get("executionArn")
                    print(f"{Fore.GREEN}Execution ARN: {execution_arn}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}❌ Error starting execution: {e}{Style.RESET_ALL}")
                    continue
                
                sleep(2)  # Wait for execution to start before checking status
                next_token = ""
                latest_event_id = 0
                try:
                    print(f"{Fore.YELLOW}Interrupt via Ctrl + C{Style.RESET_ALL}")
                    while True:
                        if next_token:
                            response = self.SFN_client.get_execution_history(
                                executionArn=execution_arn,
                                maxResults=5,
                                nextToken=next_token
                            )
                        else:
                            response = self.SFN_client.get_execution_history(
                                executionArn=execution_arn,
                                maxResults=5
                            )
                        events = response.get("events", [])
                        event_type = "RUNNING"
                        for event in events:
                            event_type = event.get("type", "")
                            event_id = event.get("id", "")
                            if event_id <= latest_event_id:
                                continue
                            latest_event_id = event_id
                            details = next((event.get(k) for k in event if k.endswith("EventDetails")), None)
                            
                            if event_type.endswith("Scheduled"):
                                print(f"Event ID: {event_id}, Type: {event_type}")
                                if details and "input" in details:
                                    input_details = json.loads(details.get("input") or '{}')
                                    input_details.pop("ReturnValueRange", None)
                                    label = "  └──Input:"
                                    print(label)
                                    json_str = json.dumps(input_details, indent=4)
                                    indent = " " * (len(label))
                                    print('\n'.join(indent + line for line in json_str.splitlines()))
                            elif event_type.endswith("Succeeded"):
                                print(f"{Fore.GREEN}Event ID: {event_id}, Type: {event_type}{Style.RESET_ALL}")
                                if details and "output" in details:
                                    output_details = json.loads(details.get("output", '{}'))
                                    label = "  └──Output:"
                                    print(label)
                                    json_str = json.dumps(output_details, indent=4)
                                    indent = " " * (len(label))
                                    print('\n'.join(indent + line for line in json_str.splitlines()))
                            elif event_type.endswith("Failed"):
                                print(f"{Fore.RED}Event ID: {event_id}, Type: {event_type}{Style.RESET_ALL}")
                                if details and "cause" in details:
                                    cause = json.loads(details.get('cause', '{}'))
                                    print(f"  ├──Error Type: {cause.get('errorType', '')}")
                                    print(f"  └──Error Message: {cause.get('errorMessage', '')}")
                            else:
                                print(f"Event ID: {event_id}, Type: {event_type}")

                        if event_type in ["ExecutionSucceeded", "ExecutionFailed"]:
                            break

                        next_token = response.get("nextToken", "")
                        sleep(2) 

                except KeyboardInterrupt:
                    print(f"{Fore.YELLOW}\nExecution interrupted by user.{Style.RESET_ALL}")
                    self.SFN_client.stop_execution(executionArn=execution_arn, error="UserStoppedExecution", cause="Execution interrupted by user")

                except Exception as e:
                    print(f"{Fore.RED}❌ Error checking execution status: {e}{Style.RESET_ALL}")
                    break
            else:
                break

    def _random_choose_execution_input(self, execution_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomly choose execution input from value range of variables.

        Args:
            execution_input: Dictionary of execution input variables

        Returns:
            Dict[str, Any]: Randomly chosen execution input
        """
        chosen_input = {}
        for key, value in execution_input.items():
            if isinstance(value, list):
                chosen_input[key] = random.choice(value)
            elif isinstance(value, dict):
                chosen_input[key] = self._random_choose_execution_input(value)
            else:
                chosen_input[key] = value
        return chosen_input
    
    def delete_state_machine(self, state_machine_arn: str):
        """
        Delete the specified state machine.

        Args:
            state_machine_arn: ARN of the state machine to delete
        """
        delete_choice = self.get_user_input("Would you like to delete this deployed state machine? (y/n):", "text")
        if delete_choice == 'y':
            try:
                self.SFN_client.delete_state_machine(stateMachineArn=state_machine_arn)
                print(f"{Fore.GREEN}State machine: {state_machine_arn} deleted successfully!{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}❌ Error deleting state machine: {e}{Style.RESET_ALL}")

    def handle_post_execution_options(self) -> Optional[str]:
        """
        Handle options after execution completion or failure.

        Returns:
            str: Next action ('back' for Phase 2, 'restart' for Phase 1, None for exit)
        """
        print(f"\n{Fore.YELLOW}Would you like to:{Style.RESET_ALL}")
        print(f"1. Start a new workflow (return to Phase 1)")
        print(f"2. Refine current workflow (return to Phase 2)")
        print(f"3. Exit (workflow complete)")

        choice = self.get_user_input("Your choice (1/2/3):", "text")

        if choice == "1":
            return "restart"
        elif choice == "2":
            return "back"
        else:
            return None  # Exit