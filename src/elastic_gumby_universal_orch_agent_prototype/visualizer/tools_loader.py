"""
Tools Loader

This module provides functionality to load and process tool definitions
from various sources including JSON input, files, and other formats.
Integrated from phase1_tools_onboarding.py for better modularity.
"""

import json
from typing import Any, Dict, List, Union

from .base import Colors


class ToolsLoader:
    """Handles loading and processing of tool definitions from various sources."""

    def __init__(self, use_colors: bool = True):
        """Initialize the tools loader.

        Args:
            use_colors: Whether to use colored output for messages
        """
        self.use_colors = use_colors

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text

    def _print_status(self, message: str, color: str = Colors.WHITE) -> None:
        """Print a status message with optional coloring."""
        print(self._colorize(message, color))

    def load_tools_from_json_string(self, json_input: str) -> Dict[str, Any]:
        """
        Load processed tools from JSON string input.

        Args:
            json_input: JSON string containing tool definitions

        Returns:
            Dict containing:
                - 'success': bool indicating if loading was successful
                - 'tools': List of loaded tools (empty if failed)
        """
        result = {"success": False, "tools": []}

        try:
            data = json.loads(json_input.strip())

            # Extract tools list from different possible structures
            tools_list = self._extract_tools_from_data(data)

            validation_result = self._validate_tools_collection(tools_list)
            if not validation_result:
                return result

            result.update({"success": True, "tools": tools_list})

            self._print_status(
                f"Successfully loaded {len(tools_list)} tool(s) from JSON", Colors.GREEN
            )
            return result

        except json.JSONDecodeError as e:
            self._print_status(f"✘ Invalid JSON format: {e}", Colors.RED)
            return result
        except Exception as e:
            self._print_status(f"✘ Error loading tools from JSON: {e}", Colors.RED)
            return result

    def _extract_tools_from_data(
        self, data: Union[Dict[str, Any], List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract tools list from various JSON data structures.

        Args:
            data: Parsed JSON data

        Returns:
            List of tool dictionaries
        """
        if isinstance(data, list):
            # Direct list of tools
            return data

        elif isinstance(data, dict):
            # Check for common container structures
            if "available_tools" in data:
                return data["available_tools"]
            elif "tools" in data:
                return data["tools"]
            elif "tool_definitions" in data:
                return data["tool_definitions"]
            elif "name" in data and "description" in data:
                # Single tool definition
                return [data]

        return []

    def _validate_tools_collection(self, tools: List[Dict[str, Any]]):
        """
        Validate a collection of tools.

        Args:
            tools: List of tool dictionaries to validate

        Returns:
            Bool indicating if all tools are valid
        """
        if not tools:
            self._print_status("✘ No tools provided for validation", Colors.RED)
            return False

        overall_valid = True

        for i, tool in enumerate(tools):
            validation = self._validate_tool_structure(tool)
            tool_name = tool.get("name", f"Tool_{i+1}")
            if validation["valid"]:
                self._print_status(f"✓ Tool {i+1}: {tool_name} is valid", Colors.GREEN)
            else:
                overall_valid = False
                self._print_status(f"✘ Tool {i+1}: {tool_name} has issues", Colors.RED)
                for error in validation["errors"]:
                    self._print_status(f"  - {error}", Colors.YELLOW)

        return overall_valid

    def _validate_tool_structure(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that a tool has the required structure.

        Args:
            tool: Tool dictionary to validate

        Returns:
            Dict containing:
                - 'valid': bool indicating if tool is valid
                - 'errors': List of validation errors
                - 'warnings': List of validation warnings
        """
        # Create typed lists to avoid mypy issues
        error_list: List[str] = []

        # Create result dictionary with typed references
        validation_result: Dict[str, Any] = {"valid": True, "errors": error_list}

        # Check required fields
        required_fields = ["name", "description", "resource", "parameters", "return"]
        for field in required_fields:
            if field not in tool:
                error_list.append(f"Missing required field: {field}")
                validation_result["valid"] = False

        # Validate parameters structure if present
        if "parameters" in tool:
            if not isinstance(tool["parameters"], list):
                error_list.append("Parameters should be a list")
                validation_result["valid"] = False
            else:
                for i, param in enumerate(tool["parameters"]):
                    try:
                        self._validate_parameter(param, i, "Parameter")
                    except ValueError as ve:
                        error_list.append(str(ve))
                        validation_result["valid"] = False

        if "return" in tool:
            try:
                self._validate_parameter(tool["return"], 0, "Return")
            except ValueError as ve:
                error_list.append(str(ve))
                validation_result["valid"] = False

        return validation_result

    def _validate_parameter(self, param: Dict[str, Any], index: int, structure_type: str) -> None:
        """
        Validate a single parameter / return structure.

        Args:
            param: structure dictionary to validate
            index: Index of the parameter in the list

        Raises:
            ValueError: If parameter structure is invalid
        """
        if not isinstance(param, dict):
            raise ValueError(f"{structure_type} at index {index} should be a dictionary")

        required_fields = ["name", "type", "description"] if structure_type == "Parameter" else ["type", "description"]
        for field in required_fields:
            if field not in param:
                raise ValueError(
                    f"{structure_type} at index {index} missing required field: {field}"
                )
