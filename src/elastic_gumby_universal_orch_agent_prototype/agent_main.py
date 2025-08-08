"""
Universal Transformation Orchestration Agent (UTOA) - Main Interface 

This is the main entrance point for the UTOA system that guides users through
the three-phase workflow orchestration process:
1. Tools Onboarding
2. Planning and Reflecting
3. Transform Execution

The interface provides interactive guidance and handles user input through
text input and file input methods.
"""

import json
from prompt_toolkit import prompt, styles
from datetime import datetime, timezone
from pathlib import Path

from colorama import Fore, Style

from elastic_gumby_universal_orch_agent_prototype.phases.phase1_tools_onboarding import Phase1ToolsOnboarding
from elastic_gumby_universal_orch_agent_prototype.phases.phase2_planning_reflecting import Phase2PlanningReflecting
from elastic_gumby_universal_orch_agent_prototype.phases.phase3_transform_execution import Phase3TransformExecution
from elastic_gumby_universal_orch_agent_prototype.visualizer.tools_visualizer import ToolsVisualizer
from elastic_gumby_universal_orch_agent_prototype.visualizer.workflow_visualizer import WorkflowVisualizer


class AgentMainInterface:
    """
    Main interface for the Universal Transformation Orchestration Agent.

    This class orchestrates the three-phase workflow process and provides
    interactive guidance to users throughout the entire workflow lifecycle.
    """

    def __init__(self):
        """Initialize the UTOA main interface."""
        self.current_phase = 1
        self.session_id = self._generate_session_id()
        self.session_data = {
            "session_id": self.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "current_phase": self.current_phase,
            "phase_history": [],
            "tools": [],  # Store tools as a dictionary for easier access
            "claude_messages": [],  # Store Claude conversation history
        }

        # Create session directory
        self.session_dir = Path(f"sessions/{self.session_id}")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.tools_visualizer = ToolsVisualizer()
        self.workflow_visualizer = WorkflowVisualizer()

        self.prompt_style = styles.Style.from_dict({
            "prompt": "ansimagenta bold",
        })

        # Initialize phase handlers
        self.phase1_handler = Phase1ToolsOnboarding(
            self.session_data, self._get_user_input, self.tools_visualizer
        )

        self.phase2_handler = Phase2PlanningReflecting(
            self.session_data, self._get_user_input, self.workflow_visualizer
        )

        self.phase3_handler = Phase3TransformExecution(
            self.session_data, self._get_user_input, self.session_dir
        )

        print(f"{Fore.GREEN}🚀 UTOA Session {self.session_id} initialized{Style.RESET_ALL}")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"utoa_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    def _print_banner(self):
        """Print the UTOA welcome banner."""
        banner = f"""
{Fore.CYAN} 
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        Universal Transformation Orchestration Agent (UTOA)                   ║
║                                                                              ║
║    Automate complex system migrations and transformation workflows           ║
║    through intelligent discovery, planning, and execution.                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.YELLOW}Session ID: {self.session_id}{Style.RESET_ALL}
{Fore.YELLOW}Session Directory: {self.session_dir}{Style.RESET_ALL}
"""
        print(banner)

    def _get_user_input(self, prompt: str, input_type: str = "text", allow_done: bool = False) -> str:
        """
        Get user input with enhanced editing capabilities.

        Args:
            prompt: The prompt to display to the user
            input_type: Type of input expected ("text", "file", "multiline")
            allow_done: Whether to allow "done" as a special command

        Returns:
            User input as string
        """
        print(f"{Fore.YELLOW}{prompt}{Style.RESET_ALL}")

        if input_type == "file":
            print(f"Options:")
            print(f"1. Type input directly line by line with Enter on empty line to finish")
            print(f"2. Provide a file path (starting with 'file:')")
            if allow_done:
                print(f"3. Type 'done' to finish this phase")
                print(f"4. Type 'quit' to exit")
            else:
                print(f"3. Type 'quit' to exit")

            user_input = self._editor(multiline=False).strip()

            if "file:" in user_input:
                file_path = user_input.split("file:", 1)[1].strip()
                try:
                    # If the right-hand side of / is an absolute path, it will ignore the left-hand side and just return the absolute path.
                    with open(Path.cwd() / file_path, "r", encoding="utf-8") as f:
                        return f.read().strip()
                except Exception as e:
                    print(f"{Fore.RED}Error reading file: {e}{Style.RESET_ALL}")
                    return self._get_user_input(prompt, input_type, allow_done)
            else:
                return user_input.lower()

        elif input_type == "multiline":
            return self._editor(multiline=True)

        else:
            return self._editor(multiline=False)
    
    def _editor(self, multiline = False) -> str:
        """
        Simple multiline input - Enter on empty line to finish.

        Returns:
            User input as string
        """
        lines: list[str] = []
        while True:
            try:
                line = prompt([('class:prompt', '> ')], style=self.prompt_style).strip()
                if not multiline:
                    return line
                elif line == "" and lines:
                    return "\n".join(lines).strip()
                lines.append(line)

            except (EOFError, KeyboardInterrupt):
                print(f"\n{Fore.YELLOW}Input cancelled.{Style.RESET_ALL}")
                return "quit"

    def _save_session_data(self):
        """Save current session data to file and generate visualization files."""
        session_file = self.session_dir / "session_data.json"
        try:
            # Update session metadata
            self.session_data["last_updated"] = datetime.now(timezone.utc).isoformat()
            self.session_data["current_phase"] = self.current_phase

            # Extract and save claude_messages and visualization data
            self._save_visualization()
            self._save_claude_messages()

            # Save main session data JSON (without claude_messages to avoid duplication)
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
            print(f"{Fore.GREEN}✅ Session data saved to {session_file}{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}❌ Error saving session data: {e}{Style.RESET_ALL}")

    def _save_visualization(self):
        """Generate and save visualizations from session data."""
        try:
            # Save tools and visualization
            tools = self.session_data.get("tools", [])
            if tools:
                tools_file = self.session_dir / "tools.json"
                tools_visualization_file = self.session_dir / "tools_visualization.md"
                with open(tools_file, "w", encoding="utf-8") as f:
                    json.dump(tools, f, indent=2, ensure_ascii=False)
                print(f"{Fore.GREEN}✅ Tools saved to {tools_file}{Style.RESET_ALL}")
                self.tools_visualizer.save_tools_visualization(tools, tools_visualization_file)
                del self.session_data["tools"]  # Remove tools to avoid duplication in session_data.json
            else:
                print(f"{Fore.YELLOW}⚠️  No tools to save and visualization{Style.RESET_ALL}")

            # Save workflow and visualization
            workflow_plan = self.session_data.get("workflow_plan", {})
            if workflow_plan:
                workflow_file = self.session_dir / "workflow.json"
                workflow_visualization_file = self.session_dir / "workflow_visualization.md"
                with open(workflow_file, "w", encoding="utf-8") as f:
                    json.dump(workflow_plan, f, indent=2, ensure_ascii=False)   
                print(f"{Fore.GREEN}✅ Workflow saved to {workflow_file}{Style.RESET_ALL}")
                self.workflow_visualizer.save_workflow_visualization(workflow_plan, workflow_visualization_file)
                del self.session_data["workflow_plan"]  # Remove workflow to avoid duplication in session_data.json
            else:
                print(f"{Fore.YELLOW}⚠️  No workflow plan to save and visualization{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}❌ Error saving visualizations: {e}{Style.RESET_ALL}")

    def _save_claude_messages(self):
        """Extract claude_messages from session_data and save to separate file."""
        try:
            claude_messages = self.session_data.get("claude_messages", [])

            if not claude_messages:
                print(f"{Fore.YELLOW}⚠️  No Reasoning history to save{Style.RESET_ALL}")
                return

            claude_messages_file = self.session_dir / "reasoning_history.json"

            # Prepare claude messages data with metadata
            total_conversations = len(claude_messages)
            claude_messages_data = {
                "session_id": self.session_id,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "total_conversations": total_conversations,
            }

            # Calculate total messages across all conversations
            total_interactions = 0
            for conversation in claude_messages:
                if isinstance(conversation, dict) and "interaction_count" in conversation:
                    total_interactions += conversation["interaction_count"]

            claude_messages_data["total_interactions"] = total_interactions
            claude_messages_data["raw_messages"] = claude_messages

            # Save claude messages to separate file
            with open(claude_messages_file, "w", encoding="utf-8") as f:
                json.dump(claude_messages_data, f, indent=2, ensure_ascii=False)

            print(
                f"{Fore.GREEN}✅ Reasoning history saved to {claude_messages_file} ({total_conversations} conversations, {total_interactions} total interactions){Style.RESET_ALL}"
            )

            # Clean claude_messages from session_data to avoid duplication in session_data.json
            if "claude_messages" in self.session_data:
                del self.session_data["claude_messages"]

        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Could not save reasoning history: {e}{Style.RESET_ALL}")

    def _record_phase_transition(self, from_phase: int, to_phase: int, reason: str = ""):
        """Record phase transition in session history."""
        transition = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_phase": from_phase,
            "to_phase": to_phase,
            "reason": reason,
        }
        self.session_data["phase_history"].append(transition)

    def run(self):
        """Main execution loop for the UTOA interface."""
        self._print_banner()

        try:
            while True:
                if self.current_phase == 1:
                    result = self.phase1_handler.run()

                    if result:
                        self._record_phase_transition(1, 2, "Phase 1 completed successfully")
                        self.current_phase = 2
                    else:
                        break  # User wants to exit

                elif self.current_phase == 2:
                    result = self.phase2_handler.run()

                    if result == "next":
                        self._record_phase_transition(2, 3, "Workflow plan approved")
                        self.current_phase = 3
                    elif result == "back":
                        self._record_phase_transition(2, 1, "User requested return to Phase 1")
                        self.current_phase = 1
                    else:
                        break  # User wants to exit

                elif self.current_phase == 3:
                    result = self.phase3_handler.run()

                    if result == "back":
                        self._record_phase_transition(3, 2, "User requested workflow refinement")
                        self.current_phase = 2
                    elif result == "restart":
                        self._record_phase_transition(3, 1, "User requested new workflow")
                        self.current_phase = 1
                    else:
                        break  # Successful completion or exit

                else:
                    print(f"{Fore.RED}Invalid phase: {self.current_phase}{Style.RESET_ALL}")
                    break

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}🛑 Process interrupted by user{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
        finally:
            self._save_session_data()
            self._print_farewell()

    def _print_farewell(self):
        """Print farewell message with session summary."""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}👋 Thank you for using UTOA!")
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}Session Summary:")
        print(f"{Fore.CYAN}  • Session ID: {self.session_id}")
        print(f"{Fore.CYAN}  • Final Phase: {self.current_phase}")
        print(
            f"{Fore.CYAN}  • Phase Transitions: {len(self.session_data.get('phase_history', []))}"
        )

        # Display tools count
        tools_count = len(self.session_data.get("tools", {}))
        print(f"{Fore.CYAN}  • Tools Processed: {tools_count}")

        print(f"{Fore.CYAN}  • Session Data: {self.session_dir}")

        # Display phase history if available
        if self.session_data.get("phase_history"):
            print(f"{Fore.CYAN}\nPhase History:")
            for i, transition in enumerate(self.session_data["phase_history"], 1):
                print(
                    f"{Fore.CYAN}  {i}. Phase {transition['from_phase']} → {transition['to_phase']}: {transition['reason']}"
                )

        # Display saved files information
        print(f"{Fore.CYAN}\nSaved Files:")
        print(f"{Fore.CYAN}  • session_data.json - Main session data")
        print(f"{Fore.CYAN}  • reasoning_history.json - Claude reasoning history")
        print(f"{Fore.CYAN}  • tools_visualization.md - Tools visualization")
        print(f"{Fore.CYAN}  • workflow_visualization.md - Workflow visualization")

        print(
            f"{Style.RESET_ALL}\n{Fore.GREEN}All session data has been saved for future reference.{Style.RESET_ALL}"
        )
