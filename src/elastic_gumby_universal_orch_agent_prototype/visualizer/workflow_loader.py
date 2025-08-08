"""
Workflow Loader

This module provides functionality to load and process workflow definitions
from various sources including JSON files, test cases, and other formats.
Designed to make workflow_visualizer.py more testable and modular.
"""

import json
import re
from typing import Any, Dict, List, Set

from .base import Colors

class WorkflowLoader:
    """Handles loading and processing of workflow definitions from various sources."""

    def __init__(self, use_colors: bool = True, tools_definition: Dict[str, Any] = None):
        """Initialize the workflow loader.

        Args:
            use_colors: Whether to use colored output for messages
            tools_definition: Dictionary containing available tools and their parameter definitions
        """
        self.use_colors = use_colors
        self.tools_definition: Dict[str, Any] = tools_definition

    def _colorize(self, text: str, color: str) -> str:
        """Apply color formatting to text if colors are enabled.
        
        Args:
            text: Text to colorize
            color: Color code to apply
            
        Returns:
            Colorized text if use_colors is True, otherwise original text
        """
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text

    def _print_status(self, message: str, color: str = Colors.WHITE) -> None:
        """Print a status message with optional color formatting.
        
        Args:
            message: Status message to print
            color: Color code to apply (defaults to white)
        """
        print(self._colorize(message, color))

    def load_workflow_from_json_string(self, json_string: str) -> Dict[str, Any]:
        """
        Load and validate workflow from a JSON string with comprehensive error handling.

        Parses JSON string, extracts workflow structure, normalizes the data,
        validates the workflow, and provides detailed feedback with colored output.

        Args:
            json_string: JSON string containing workflow data

        Returns:
            Dict with validation result containing:
            - success: Boolean indicating if workflow was successfully loaded and validated
            - workflow: Normalized workflow structure if successful, None otherwise
            - errors: List of error messages encountered during processing
        """
        result = {"success": False, "workflow": None, "errors": []}
        try:
            data = json.loads(json_string.strip())
            result['workflow'] = self._extract_workflow_from_data(data)
            result['workflow'] = self._normalize_workflow_structure(result['workflow'])

        except json.JSONDecodeError as e:
            self._print_status(f"✘ Invalid JSON format: {e}", Colors.RED)
            result['errors'].append(f"Invalid JSON format: {e}")
            return result
        except ValueError as e:
            self._print_status(f"✘ Error extracting workflow: {e}", Colors.RED)
            result['errors'].append(f"Error extracting workflow: {e}")
            return result

        validation_result = self.validate_workflow(result['workflow'])
        if validation_result["is_valid"]:
            result["success"] = True  # Update success flag
            self._print_status(
                f"✓ Successfully loaded workflow: {result['workflow']['name']}\n"
                f"  Total nodes: {validation_result['node_count']}\n"
                f"  Node types: {', '.join(sorted(validation_result['node_types']))}", Colors.GREEN
            )
        else:
            errors_str = '\n'.join(validation_result['errors'])
            self._print_status(
                f"✘ Workflow validation failed:\n{errors_str}", Colors.RED
            )
            result['errors'].extend(validation_result['errors'])
        return result

    def _extract_workflow_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract workflow structure from loaded JSON data supporting multiple formats.

        Handles two main formats:
        1. Direct workflow format with root, name, and description fields
        2. Test case format with expected_workflow containing the workflow structure

        Args:
            data: Loaded JSON data dictionary

        Returns:
            Dict containing the extracted workflow structure

        Raises:
            ValueError: If no valid workflow structure is found in the data
        """
        def _is_valid_workflow(data: Dict[str, Any]) -> bool:
            """Check if data has required workflow fields."""
            return isinstance(data, dict) and all(field in data for field in ["root", "name", "description"])

        workflow = None

        # Case 1: Direct workflow format
        if _is_valid_workflow(data):
            workflow = data

        # Case 2: Test case format with expected_workflow
        elif "expected_workflow" in data and _is_valid_workflow(data["expected_workflow"]):
            workflow = data["expected_workflow"]

        if workflow is None:
            raise ValueError("No valid workflow structure found in the provided data")

        return workflow
    
    def _normalize_workflow_structure(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize workflow structure ensuring all properties are proper dictionaries.

        Converts JSON string properties to dictionaries and recursively normalizes
        the root node structure. Handles cases where 'root' might be a JSON string
        instead of a dictionary object.

        Args:
            workflow: Raw workflow dictionary potentially containing JSON strings

        Returns:
            Dict with normalized workflow structure where all nested objects are dictionaries

        Raises:
            json.JSONDecodeError: If JSON string parsing fails
            ValueError: If 'root' property is not a dictionary or valid JSON string
        """
        # Ensure 'root' is a dictionary, not a JSON string
        root = workflow["root"]
        if isinstance(root, str):
            try:
                root = json.loads(root)
            except json.JSONDecodeError as e:
                self._print_status(f"✘ Invalid JSON in 'root': {e}", Colors.RED)
                raise e
        elif not isinstance(root, dict):
            raise ValueError(
                f"'root' property must be a dictionary or valid JSON string, got {type(root)}"
            )

        workflow["root"] = self._normalize_node_structure(root)

        return workflow

    def _normalize_node_structure(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively normalize node structures ensuring proper dictionary types.

        Converts JSON string properties to dictionaries for common node properties
        (parameters, outputVariable, condition, onTimeout, errorHandler) and
        recursively processes nested structures like steps, branches, onTimeout,
        and errorHandler. Follows immutability principle by creating a copy.

        Args:
            node: Node dictionary to normalize

        Returns:
            Dict with normalized node structure where JSON strings are converted to dictionaries

        Raises:
            json.JSONDecodeError: If JSON string parsing fails for any property
        """
        normalized = node.copy() # follows the principle of immutability 

        # Handle common node properties that might be JSON strings
        json_properties = ["parameters", "outputVariable", "condition", "onTimeout", "errorHandler"]

        for prop in json_properties:
            if prop in normalized and isinstance(normalized[prop], str):
                try:
                    # Try to parse as JSON if it looks like JSON
                    prop_value = normalized[prop].strip()
                    if prop_value.startswith(("{", "[")):
                        normalized[prop] = json.loads(prop_value)
                except json.JSONDecodeError as e:
                    self._print_status(f"✘ Invalid JSON in '{prop}': {normalized[prop]}", Colors.RED)
                    raise e
                
        # Recursively handle nested structures
        if "steps" in normalized and isinstance(normalized["steps"], list):
            normalized["steps"] = [
                self._normalize_node_structure(step) if isinstance(step, dict) else step
                for step in normalized["steps"]
            ]

        if "branches" in normalized and isinstance(normalized["branches"], list):
            normalized["branches"] = [
                self._normalize_node_structure(branch) if isinstance(branch, dict) else branch
                for branch in normalized["branches"]
            ]

        if "onTimeout" in normalized and isinstance(normalized["onTimeout"], dict):
            normalized["onTimeout"] = self._normalize_node_structure(normalized["onTimeout"])

        if "errorHandler" in normalized and isinstance(normalized["errorHandler"], dict):
            normalized["errorHandler"] = self._normalize_node_structure(normalized["errorHandler"])

        return normalized

    def validate_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate workflow structure and collect comprehensive metadata.

        Performs structural validation checking for required properties (name, description, root),
        recursively validates all nodes, collects node types and counts, and validates
        variable references and tool definitions.

        Args:
            workflow: Workflow dictionary to validate

        Returns:
            Dict containing comprehensive validation results:
            - is_valid: Boolean indicating if workflow passed all validations
            - node_count: Total number of nodes in the workflow
            - node_types: Set of unique node types found in the workflow
            - errors: List of validation error messages
        """
        errors: List[str] = []
        node_types: Set[str] = set()

        validation_result: Dict[str, Any] = {
            "is_valid": True,
            "node_count": 0,
            "node_types": node_types,
            "errors": errors
        }

        try:
            # Check basic structure
            if not isinstance(workflow, dict):
                errors.append("Workflow must be a dictionary")
                validation_result["is_valid"] = False
                return validation_result

            # Check for name
            if "name" in workflow:
                validation_result["is_valid"] &= isinstance(workflow["name"], str)
            else:
                errors.append("Workflow must have a 'name' property")

            # Check for description
            if "description" in workflow:
                validation_result["is_valid"] &= isinstance(workflow["description"], str)
            else:
                errors.append("Workflow must have a 'description' property")
    
            # Check for root structure
            if "root" in workflow:
                validation_result["is_valid"] &= isinstance(workflow["root"], dict)
                self._validate_node_content(workflow["root"], validation_result, path="root", defined_vars=set())
            else:
                errors.append("Workflow must have a 'root' property")

            validation_result["is_valid"] &= (len(errors) == 0 and validation_result["node_count"] > 0)

        except Exception as e:
            errors.append(f"Workflow validation error: {e}")
            validation_result["is_valid"] = False

        return validation_result

    def _validate_node_content(self, node: Dict[str, Any], validation_result: Dict[str, Any], path: str, defined_vars: Set[str]) -> None:
        """
        Recursively validate node content, count nodes, collect types, and track variable scope.

        Validates node structure, conditions, parameters, variable references, and
        recursively processes child nodes (errorHandler, ifTrue/ifFalse, body, steps, branches).
        Maintains variable scope tracking to ensure variables are defined before use.
        Updates validation_result with node counts, types, and error messages.

        Args:
            node: Current node dictionary to validate
            validation_result: Validation result dictionary to update with counts, types, and errors
            path: Current path in the workflow for detailed error reporting
            defined_vars: Set of variable names defined in current execution context
        """
        # Count this node and its type
        validation_result["node_count"] += 1
        validation_result["node_types"].add(node["type"])
        path += f".{node['type']}" 

        if "condition" in node:
            if node["condition"]["type"] == "comparison":
                self._validate_condition(node["condition"], validation_result, path + ".condition", defined_vars)
            elif node["condition"]["type"] == "logical":
                for condition in node["condition"]["conditions"]:
                    self._validate_condition(condition, validation_result, path + ".condition.conditions", defined_vars)

        if "parameters" in node:
            self._validate_parameters(node, validation_result, path + ".parameters", defined_vars)
        
        if "outputVariable" in node:
            defined_vars.add(node["outputVariable"])

        # Process children for tool_call
        if "errorHandler" in node:
            self._validate_node_content(node["errorHandler"], validation_result, path + ".errorHandler", defined_vars)
        
        # Process children for branch
        if "ifTrue" in node:
            self._validate_node_content(node["ifTrue"], validation_result, path + ".ifTrue", defined_vars.copy())
        if "ifFalse" in node:
            self._validate_node_content(node["ifFalse"], validation_result, path + ".ifFalse", defined_vars.copy())
        
        # Process children for loop
        if "body" in node:
            self._validate_node_content(node["body"], validation_result, path + ".body", defined_vars)

        # Process children for sequence
        if "steps" in node:
            self._validate_container(node, validation_result, path + ".steps", defined_vars)
        
        if "branches" in node:   
            self._validate_container(node, validation_result, path + ".branches", defined_vars)
            
        # Process children for wait_for_event
        if "entityId" in node:
            match = re.fullmatch(r"^\{\%\s*\$([a-zA-Z_][\w\.]*)\s*\%\}$", node["entityId"])
            if not match:
                validation_result["errors"].append(
                    f"Variable reference in '{node['entityId']}' at {path}.entityId should follow {{% $varName %}} format"
                )
            elif match.group(1).split(".", 1)[0] not in defined_vars:
                validation_result["errors"].append(
                    f"Variable reference in '{node['entityId']}' at '{path}.entityId' is not defined before use in its execution context"
                )
        elif "prompt" in node:
            if "{% " in node["prompt"] and "%}" in node["prompt"]:
                if not node["prompt"].startswith("{%") or not node["prompt"].endswith("%}"):
                    validation_result["errors"].append(
                        f"Variable reference in '{node['prompt']}' at {path}.prompt should follow {{% 'Prompt guidance text' & $varName & ' end' %}} format"
                    )
                else:
                    var_names = re.findall(r"\$([a-zA-Z_][\w\.]*)", node["prompt"])
                    for var_name in var_names:
                        var_name = var_name.split(".", 1)[0]
                        if var_name not in defined_vars:
                            validation_result["errors"].append(
                                f"Variable '{var_name}' in '{node['prompt']}' at '{path}.prompt' is not defined before use in its execution context"
                            )
    
        if "onTimeout" in node:
            self._validate_node_content(node["onTimeout"], validation_result, path + ".onTimeout", defined_vars)


    def _validate_condition(self, condition, validation_result: Dict[str, Any], path: str, defined_vars: Set[str]) -> None:
        """
        Validate condition structure including operand types and variable references.

        Validates left and right operands ensuring proper format for variable references
        ({% $varName %} pattern), checks variable scope, and validates operand types.
        Left operands must be variable references, right operands can be variables or literals.

        Args:
            condition: Condition dictionary containing left, right, and operator
            validation_result: Validation result dictionary to update with errors
            path: Current path in the workflow for detailed error reporting
            defined_vars: Set of variable names defined in current execution context
        """
        single_var_ref_pattern = r"^\{\%\s*\$([a-zA-Z_][\w\.]*)\s*\%\}$"
        def _validate_variable_in_condition(operand: str, validation_result: Dict[str, Any], path: str) -> None:
            """Helper function to validate variable references in conditions."""
            match = re.fullmatch(single_var_ref_pattern, operand)
            if not match:
                if "{% " in operand and "%}" in operand:
                    if "[" in operand or "(" in operand:
                        validation_result["errors"].append(
                            f"Condition operand: '{operand}' at '{path}' should NOT contain brackets or parentheses in JSONata expression {{% ... %}}"
                        )
                    elif path.endswith(".right") and "$" not in operand:
                        validation_result["errors"].append(
                            f"Right operand with static value '{operand}' at '{path}' should be string without {{% ... %}} wrapping"
                        )
                    else:
                        validation_result["errors"].append(
                            f"Condition operand: '{operand}' at '{path}' should follow {{% $varName %}} format"
                        )
                elif path.endswith(".left"):
                    validation_result["errors"].append(
                        f"Left operand '{operand}' at '{path}' should follow {{% $varName %}} format"
                    )
                else:
                    pass # pure string right operand
            elif match.group(1).split(".", 1)[0] not in defined_vars:
                validation_result["errors"].append(
                    f"Variable '{match.group(1)}' at '{path}' is not defined before use in its execution context"
                )

        if isinstance(condition["left"], str):
            _validate_variable_in_condition(condition["left"], validation_result, path + ".left")
        elif isinstance(condition["left"], (int, float, bool)):
            validation_result["errors"].append(
                f"Invalid left operand type {type(condition['left']).__name__} at '{path}.left'"
            )
        
        if isinstance(condition["right"], str):
            _validate_variable_in_condition(condition["right"], validation_result, path + ".right")
        elif not isinstance(condition["right"], (int, float, bool)):
            validation_result["errors"].append(
                f"Invalid right operand type : {type(condition['right']).__name__} at '{path}.right'"
            )

    def _validate_parameters(self, node: Dict[str, Any], validation_result: Dict[str, Any], path: str, defined_vars: Set[str]) -> None:
        """
        Validate tool parameters including tool existence, parameter names, and variable references.

        Checks if the tool exists in tools_definition, validates parameter names against
        tool schema, validates parameter value types and formats, and ensures variable
        references follow proper JSONata syntax patterns. Handles nested parameter objects.

        Args:
            node: Node dictionary containing toolName and parameters
            validation_result: Validation result dictionary to update with errors
            path: Current path in the workflow for detailed error reporting
            defined_vars: Set of variable names defined in current execution context
        """
        parameter_pattern = r"^\{\%[^\[\]\(\)]*\$[a-zA-Z_][\w\.]*[^\[\]\(\)]*\%\}$"
        
        def _validate_value_in_parameters(value: Any, key: str, path: str) -> None:
            """Helper function to validate values in parameters."""
            if isinstance(value, str):
                match = re.fullmatch(parameter_pattern, value)
                if not match:
                    if "{% " in value and "%}" in value:
                        if "[" in value or "(" in value:
                            validation_result["errors"].append(
                                f"Parameter value '{key}': '{value}' at '{path}' should NOT contain brackets or parentheses in JSONata expression {{% ... %}}"
                            )
                        elif "$" not in value:
                            validation_result["errors"].append(
                                f"Parameter static value '{key}': '{value}' at '{path}' should be string without {{% ... %}} wrapping"
                            )
                        else:
                            validation_result["errors"].append(
                                f"Parameter value '{key}': '{value}' at '{path}' should follow {{% $varName1 & ' some text ' & $varName2 %}} format"
                            )
                    else: 
                        pass # pure string parameter value
                else:
                    var_names = re.findall(r"\$([a-zA-Z_][\w\.]*)", value)
                    for var_name in var_names:
                        var_name = var_name.split(".", 1)[0] # Extract variable name without sub-properties
                        if var_name not in defined_vars:
                            validation_result["errors"].append(
                                f"Variable '{var_name}' in parameter '{key}' at '{path}' is not defined before use in its execution context"
                            )
            elif not isinstance(value, (int, float, bool)) and value is not None:
                validation_result["errors"].append(
                    f"Invalid {type(value).__name__} parameter type for '{key}': '{value}' at '{path}'"
                )
            else:
                pass # Valid types: int, float, bool, None
 
        if node["toolName"] not in self.tools_definition:
            validation_result["errors"].append(
                f"'{node['toolName']}' at '{path}' does NOT exist in AVAILABLE_TOOLS"
            )
            return

        valid_parameter_names = set([para['name'] for para in self.tools_definition.get(node["toolName"])])
        for key, value in node["parameters"].items():
            if key not in valid_parameter_names:
                validation_result["errors"].append(
                    f"Invalid parameter name '{key}' for '{node['toolName']}' at '{path}'"
                )
                continue
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    _validate_value_in_parameters(sub_value, f"{key}.{sub_key}", path)
            else:
                _validate_value_in_parameters(value, key, path)
    
    def _validate_container(self, node: Dict[str, Any], validation_result: Dict[str, Any], path: str, defined_vars: Set[str]) -> None:
        """
        Validate container nodes (parallel, sequence) with specific container rules.

        Checks for empty containers, single-item containers (which should be unwrapped),
        and recursively validates child nodes. Handles variable scope differently for
        parallel branches (copied scope) vs sequential steps (shared scope).

        Args:
            node: Container node dictionary with steps or branches
            validation_result: Validation result dictionary to update with errors
            path: Current path in the workflow for detailed error reporting
            defined_vars: Set of variable names defined in current execution context
        """
        container = node.get("steps", []) or node.get("branches", [])
        if not container:
            validation_result["errors"].append(
                f"Remove empty '{node.get('type')}' container at '{path}'"
            )
        elif len(container) == 1:
            validation_result["errors"].append(
                f"Remove wrapper {node.get('type')} container with only one {container[0].get('type')} node at '{path}'"
            )
        elif path.endswith(".branches"):
            for i, sub_node in enumerate(container):
                # Copy defined_vars to avoid differnt branches affecting each other
                self._validate_node_content(sub_node, validation_result, path + f"[{i}]", defined_vars.copy())
        else:
            for i, sub_node in enumerate(container):
                # All previous defined_vars should be valid in all steps
                self._validate_node_content(sub_node, validation_result, path + f"[{i}]", defined_vars)