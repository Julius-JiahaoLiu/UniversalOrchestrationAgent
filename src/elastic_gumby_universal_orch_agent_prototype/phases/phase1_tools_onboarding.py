"""
Phase 1: Available Tools Onboarding

This module handles the collection and processing of tool descriptions
from users, converting them into structured ToolDefinition format.
"""

from typing import Any, Dict

from colorama import Fore, Style

from elastic_gumby_universal_orch_agent_prototype.transform.tool_description_transformer import ToolDescriptionTransformer


class Phase1ToolsOnboarding:
    """
    Handles Phase 1: Available Tools Onboarding

    Collects tool descriptions from users and processes them into
    structured ToolDefinition format according to input_schema.json.
    """

    def __init__(self, session_data: Dict[str, Any], get_user_input_func, tools_visualizer):
        """
        Initialize Phase 1 handler.

        Args:
            session_data: Reference to main session data dictionary
            get_user_input_func: Function to get user input
        """
        self.session_data = session_data
        self.get_user_input = get_user_input_func
        self.tools_visualizer = tools_visualizer
        self.tools_transformer = ToolDescriptionTransformer()
    
    def run(self) -> bool:
        """
        Execute Phase 1: Available Tools Onboarding.

        Returns:
            bool: True to continue to next phase, False to exit
        """
        self.print_phase_header()
        self.print_detailed_guidance()

        # Collect tool descriptions
        if not self.collect_tool_descriptions():
            return False

        # Handle post-processing options
        return self.handle_post_processing_options()

    def print_phase_header(self):
        """Print the Phase 1 header and introduction."""
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}🔧 PHASE 1: AVAILABLE TOOLS ONBOARDING")
        print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
        print(f"\n{Fore.WHITE}In this phase, we'll identify and catalog all available tools,")
        print(f"that can be third party services, invoke functions, hundreds of AWS service endpoints used in your workflow orchestration.{Style.RESET_ALL}")

    def print_detailed_guidance(self):
        """Print detailed guidance for tool description format."""
        print(f"\n{Fore.WHITE}We'll collect detailed information about your available tools to create structured tool definitions.")
        print(f"Each tool description should include the following information:")
        print(f"\n{Fore.CYAN}🔧 Required Information for Each Tool:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  1. Tool Name:{Style.RESET_ALL} Clear, descriptive name for the tool")
        print(f"{Fore.YELLOW}  2. Description:{Style.RESET_ALL} Detailed explanation of what the tool does and its purpose")
        print(f"{Fore.YELLOW}  3. Resource:{Style.RESET_ALL} Amazon Resource Name for the tool resource")
        print(f"{Fore.YELLOW}  4. Parameters:{Style.RESET_ALL} Input parameters the tool accepts, including:")
        print(f"     • Parameter name and data type (string, number, boolean, object, array)")
        print(f"     • Description of each parameter's purpose")
        print(f"     • Whether each parameter is required or optional")
        print(f"     • Default values (if any)")
        print(f"     • Constraints (min/max values, allowed values, patterns)")
        print(f"{Fore.YELLOW}  5. Return Value:{Style.RESET_ALL} What the tool return, including:")
        print(f"     • Return value name and data type")
        print(f"     • Description of the return value")
        print(f"     • Structure/schema for complex return types")

        print(f"\n{Fore.GREEN}{'='*60}")
        print(f"{Fore.GREEN}📋 EXAMPLE TOOL DESCRIPTION")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")

        # Tool header with fancy styling
        print(f"\n{Fore.CYAN}└── 🛠️  {Style.BRIGHT}get_weather{Style.RESET_ALL}")
        print(f"{Fore.WHITE}    Retrieves current weather information for a specified location{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}    Resource: arn:aws:lambda:us-west-2:123456789012:function:get_weather{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}    📊 Parameters (2):{Style.RESET_ALL}")
        print(f"{Fore.WHITE}        ├── {Fore.RED}●{Style.RESET_ALL} {Style.BRIGHT}location{Style.RESET_ALL}")
        print(f"{Fore.WHITE}        │   {Fore.GREEN}Type: string{Style.RESET_ALL}")
        print(f"{Fore.WHITE}        │   City name or coordinates for weather lookup{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLACK_EX}        │   Constraints: min: 1, pattern: ^[a-zA-Z0-9\\s,.-]+${Style.RESET_ALL}")
        print(f"{Fore.WHITE}        └── {Fore.GREEN}○{Style.RESET_ALL} {Style.BRIGHT}units{Style.RESET_ALL}")
        print(f"{Fore.WHITE}            {Fore.GREEN}Type: string{Style.RESET_ALL}")
        print(f"{Fore.WHITE}            Temperature units for the weather data{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLACK_EX}            Default: celsius{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLACK_EX}            Constraints: enum: ['celsius', 'fahrenheit', 'kelvin']{Style.RESET_ALL}")
        print(f"{Fore.GREEN}    📤 Return:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}        └── {Style.BRIGHT}weather_data{Style.RESET_ALL}")
        print(f"{Fore.WHITE}            {Fore.MAGENTA}Type: object{Style.RESET_ALL}")
        print(f"{Fore.WHITE}            Comprehensive weather information including temperature, humidity, conditions, wind speed, and forecast data{Style.RESET_ALL}")

    def collect_tool_descriptions(self) -> bool:
        """
        Collect tool descriptions from user until they type 'done'.
        Handles both raw descriptions and pre-processed JSON files.

        Returns:
            bool: True if successful, False if user wants to quit
        """

        while True:
            print(f"\n{Fore.CYAN}📝 Tool Description Input{Style.RESET_ALL}")

            tool_input = self.get_user_input(
                f"Please provide the functionality of tools including required information above,\nor provide a JSON file path containing pre-processed tool definitions.",
                "file",
                allow_done=True,
            )

            if tool_input.lower() == "quit":
                return False
            elif tool_input.lower() == "done":
                print(f"{Fore.GREEN}✅ Finished collecting {len(self.session_data['tools'])} tool(s).{Style.RESET_ALL}")
                break
            elif tool_input.strip() == "":
                print(f"{Fore.YELLOW}⚠️ Empty input received. Please provide a tool description or type 'done' to finish.{Style.RESET_ALL}")
                continue

            # Check if this is a JSON file with processed tools
            result = self.tools_transformer.tools_loader.load_tools_from_json_string(tool_input)
            if result["success"]:
                self.session_data["tools"].extend(result["tools"])
            else:
                transformed_tool = self.tools_transformer.transform_description(tool_input)
                if transformed_tool:
                    self.session_data["tools"].append(transformed_tool)
                else:
                    print(f"{Fore.RED}✘ Rectify description to include all required properties, then try again{Style.RESET_ALL}")
            
            print(f"{Fore.CYAN}Current available tools: {len(self.session_data['tools'])}{Style.RESET_ALL}")

        return True

    def handle_post_processing_options(self) -> bool:
        """
        Handle user options after processing is complete.

        Returns:
            bool: True to proceed to Phase 2, False to restart Phase 1 or exit
        """
        print(f"\n{Fore.YELLOW}Would you like to:{Style.RESET_ALL}")
        print(f"1. Review processed tool definitions")
        print(f"2. Add more tool descriptions")
        print(f"3. Proceed to Phase 2 (Planning & User Reflection)")

        choice = self.get_user_input("Your choice (1/2/3):", "text")

        if choice == "1":
            tools = self.session_data.get("tools", [])
            print(self.tools_visualizer.visualize_tools(tools))
            return self.handle_post_processing_options()  # Return to options menu
        elif choice == "2":
            if not self.collect_tool_descriptions():
                return False
            return self.handle_post_processing_options()
        else:  # choice == "3" or any other input
            print(f"{Fore.GREEN}🚀 Ready to proceed to Phase 2: Planning & User Reflection{Style.RESET_ALL}")
            return True
