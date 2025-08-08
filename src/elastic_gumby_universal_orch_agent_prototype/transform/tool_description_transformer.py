from typing import Any, Dict
from colorama import Fore, Style
import json

from elastic_gumby_universal_orch_agent_prototype.data_schema import get_tools_schema
from elastic_gumby_universal_orch_agent_prototype.visualizer.tools_loader import ToolsLoader
from elastic_gumby_universal_orch_agent_prototype.planner.bedrock_client_manager import BedrockClientManager

class ToolDescriptionTransformer:
    """
    Transforms tool descriptions into structured ToolDefinition format.
    """

    def __init__(self, model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0"):
        """
        Initialize the transformer with the input schema.

        Args:
            input_schema_path: Path to the input schema JSON file.
        """
        self.model_id = model_id
        self.tool_schema = get_tools_schema()
        self.tool_validator = {
            "name": "tool_validator",
            "description": "This tool validates the transformed tool description against the input schema and provide detailed error messages",
            "input_schema": self.tool_schema,
        }
        self.bedrock_manager = BedrockClientManager()
        self.tools_loader = ToolsLoader()
        self.max_interactions = 5

    def transform_description(self, tool_description: str) -> Dict[str, Any]:
        """
        Transform a tool description into the ToolDefinition format.

        Args:
            tool_description: Raw tool description dictionary.

        Returns:
            Structured ToolDefinition dictionary.
        """
        transformed_tool = {}
        system_prompt = f"""
            You are a tool description transformer. Your task is to transform the TOOL_DESCRIPTION into the pre-defined format in input_schema of tool_validator.
            Please provide the transformed tool functionality via tool_use stop for tool_validator.
            If the required properties in input_schema are not provided in TOOL_DESCRIPTION, you should directly stop via end_trun and provide detailed error message in text.
        """ 
        messages = [
            {
                "role": "user",
                "content": f"""
                    TOOL_DESCRIPTION: {tool_description}
                """,
            }
        ]

        # Execute main planning loop
        print(f"\n{Fore.CYAN}🤖 Processing tool description...{Style.RESET_ALL}")
        interaction_count = 0
        while interaction_count < self.max_interactions:
            interaction_count += 1
            response = self.bedrock_manager.invoke_model(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=1000,
                tools=[self.tool_validator],
                model_id=self.model_id,
            )
            stop_reason = response.get("stop_reason", "")
            content_list = response.get("content", [])
            # Process response content
            if stop_reason == "tool_use":
                transformed_tool = self.process_tool_use(content_list, messages)
            elif stop_reason == "end_turn":
                if not transformed_tool:
                    text_parts = [item["text"] for item in content_list if item.get("type") == "text"]
                    full_text = "".join(text_parts).strip()
                    self._print_assistant_text(full_text, "Error Message")
                break
            else:
                print(f"{Fore.RED}Unexpected stop reason: {stop_reason}. Continuing with next interaction.{Style.RESET_ALL}")
                pass

            if transformed_tool: # Already get valid transformed tool, no need waste resource for end_turn interaction  
                break
        
        return transformed_tool
    
    def process_tool_use(self, content_list, messages):
        """
        Process tool_use content from the response.
        """
        tool_use_id = ""
        assistant_text = ""
        transformed_tool = {}

        for content_item in content_list:
            if content_item.get("type") == "text":
                text_content = content_item.get("text", "")
                assistant_text += text_content
            elif content_item.get("type") == "tool_use":
                transformed_tool = content_item.get("input", {})
                tool_use_id = content_item.get("id", "")
        
        if assistant_text:
            self._print_assistant_text(assistant_text, "Tool Description Transformation Reasoning")

        messages.append({"role": "assistant", "content": content_list})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "Transformed tool definition in tool_use input is received without errors.",
                    }
                ],
            }
        )

        result = self.tools_loader.load_tools_from_json_string(json.dumps(transformed_tool))

        if not result["success"]:
            messages[-1]["content"][0]["content"] = "Transformed tool definition in tool_use input is received with errors. Please rectify it to strictly follow input_schema of tool_validator."
            return {}
        else:
            return result["tools"][0] if result["tools"] else {}
    
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