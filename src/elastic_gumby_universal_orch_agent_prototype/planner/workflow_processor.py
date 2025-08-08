"""
Workflow Processor

Handles workflow data processing, section combination, and metadata extraction.
"""

import re
from colorama import Fore, Style


class WorkflowProcessor:
    """
    Processes workflow data including section combination, flattening, and metadata extraction.
    """

    def __init__(self):
        """Initialize the workflow processor."""
        pass

    def extract_final_metadata(self, text_response):
        """
        Extract name and description from Claude's final text response.
        Expected format: {"name": "...", "description": "..."}

        Args:
            text_response (str): The final text response from Claude

        Returns:
            dict: Dictionary with 'name' and 'description' keys, or empty dict if not found
        """
        try:
            text_response = text_response.strip()

            name_pattern = r'"name"\s*:\s*"([^"]*)"'
            desc_pattern = r'"description"\s*:\s*"([^"]*)"'

            name_match = re.search(name_pattern, text_response)
            desc_match = re.search(desc_pattern, text_response)

            if name_match and desc_match:
                return {
                    "name": name_match.group(1).strip(),
                    "description": desc_match.group(1).strip(),
                }

            print(
                f"{Fore.YELLOW}Warning: Could not extract final metadata from response{Style.RESET_ALL}"
            )
            return {}

        except Exception as e:
            print(
                f"{Fore.YELLOW}Warning: Error extracting final metadata: {e}{Style.RESET_ALL}"
            )
            return {}

    def combine_workflow_sections(self, workflow_sections):
        """
        Combine multiple workflow sections into a single cohesive workflow.

        Args:
            workflow_sections (list): List of workflow section dictionaries

        Returns:
            dict: Combined workflow plan
        """
        if not workflow_sections:
            return {"error": "No workflow sections to combine"}

        print(
            f"{Fore.BLUE}Combining {len(workflow_sections)} workflow sections...{Style.RESET_ALL}"
        )

        # Extract metadata from first section
        first_section = workflow_sections[0]["workflow_plan"]
        combined_name = first_section.get("name", "Combined Workflow")
        combined_description = first_section.get("description", "Multi-section workflow execution")

        # Collect all steps from all sections
        all_steps = []
        for i, section in enumerate(workflow_sections):
            section_plan = section["workflow_plan"]
            section_number = section.get("section_number", i + 1)

            if "root" not in section_plan:
                print(
                    f"{Fore.YELLOW}Warning: Section {section_number} missing 'root' element{Style.RESET_ALL}"
                )
                continue

            # flattened_steps = self.flatten_workflow_section(section_plan["root"], section_number)
            all_steps.append(section_plan["root"])

        # Create combined workflow - simplified logic
        combined_root = (
            all_steps[0]
            if len(all_steps) == 1
            else {
                "type": "sequence",
                "name": "Combined Workflow Execution",
                "description": "Sequential execution of all workflow sections",
                "steps": all_steps,
            }
        )

        combined_workflow = {
            "name": combined_name,
            "description": combined_description,
            "root": combined_root,
        }

        print(f"{Fore.GREEN}✓ Successfully combined workflow sections{Style.RESET_ALL}")
        return combined_workflow

    def flatten_workflow_section(self, workflow_node, section_number):
        """
        Flatten a workflow section node into a list of executable steps.

        Args:
            workflow_node (dict): The workflow node to flatten
            section_number (int): Section number for naming

        Returns:
            list: List of flattened workflow steps
        """
        if not isinstance(workflow_node, dict):
            return []

        node_type = workflow_node.get("type")

        # Handle container types
        if node_type == "sequence":
            return self._flatten_sequence_steps(workflow_node, section_number)
        elif node_type == "parallel":
            return self._flatten_parallel_branches(workflow_node, section_number)
        else:
            # All other node types (tool_call, user_input, branch, loop, wait_for_event)
            return [workflow_node]

    def _flatten_sequence_steps(self, workflow_node, section_number):
        """Helper method to flatten sequence steps."""
        steps = workflow_node.get("steps", [])
        flattened_steps = []

        for step in steps:
            flattened_steps.extend(self.flatten_workflow_section(step, section_number))

        return flattened_steps

    def _flatten_parallel_branches(self, workflow_node, section_number):
        """Helper method to flatten parallel branches."""
        branches = workflow_node.get("branches", [])
        if not branches:
            return []

        flattened_branches = []
        for branch in branches:
            branch_steps = self.flatten_workflow_section(branch, section_number)

            # Simplify branch handling
            if len(branch_steps) == 1:
                flattened_branches.append(branch_steps[0])
            elif branch_steps:  # len > 1
                flattened_branches.append(
                    {
                        "type": "sequence",
                        "name": f"Branch from section {section_number}",
                        "description": f"Sequential steps from section {section_number}",
                        "steps": branch_steps,
                    }
                )

        # Create parallel container
        parallel_container = {
            "type": "parallel",
            "branches": flattened_branches,
            "description": workflow_node.get(
                "description", f"Parallel execution from section {section_number}"
            ),
        }

        return [parallel_container]
