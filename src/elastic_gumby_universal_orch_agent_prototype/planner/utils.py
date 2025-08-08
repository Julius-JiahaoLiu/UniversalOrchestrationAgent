"""
Planning Utilities

Utility functions for workflow planning that provide convenient interfaces
to the core IterativePlanner.iterative_planning method.
"""

import json

from colorama import Fore, Style

from .iterative_planner import IterativePlanner


def generate_plan(
    workflow_description, 
    available_tools, 
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", 
    max_interactions=20, 
    max_tokens=8000
):
    """
    Generate a structured workflow plan based on the description using LLM-guided generative planning.

    This utility function creates a IterativePlanner instance and uses its iterative_planning
    method to generate a new workflow plan from scratch.

    Args:
        workflow_description (str): Natural language description of the workflow to be structured
        available_tools (list): List of available tools
        model_id (str): The Bedrock model ID to use for planning

    Returns:
        dict: The complete structured workflow plan as a dictionary following the workflow schema,
              ready for execution by the workflow execution engine
    """
    print(
        f"\n{Fore.BLUE}Starting LLM-guided generative workflow planning...{Style.RESET_ALL}"
    )

    # Create planner core instance
    planner = IterativePlanner(model_id=model_id, max_interactions=max_interactions, max_tokens=max_tokens)

    # Define the workflow execution tool
    workflow_execution_tool = {
        "name": "execute_workflow",
        "description": "Execute a workflow plan section or complete workflow. This tool takes a structured workflow definition (following the workflow schema) as input. For iterative planning, include a 'section_update' property to update existing sections.",
        "input_schema": planner.workflow_schema,
    }

    # Intelligent system prompt that lets LLM decide the approach
    system_prompt = """
You are an AI assistant specialized in workflow orchestration and planning. You will analyze the WORKFLOW_DESCRIPTION and AVAILABLE_TOOLS to determine the best planning approach.

**ADAPTIVE PLANNING STRATEGY:**
First, analyze the WORKFLOW_DESCRIPTION and AVAILABLE_TOOLS to determine complexity:
- **Simple workflows** (few steps, straightforward logic): Generate the complete workflow in ONE interaction
- **Complex workflows** (many steps, complex logic, multiple branches): Use iterative planning with multiple sections

**TOOL_USE STOP:**
- To generate new section, use execute_workflow tool with "name", "descripion", "root" properties
- To update previous section, use execute_workflow tool with "name", "descripion", "root" and "section_update" properties
- Always include "root" property with the structured workflow definition in input_schema

**FOR SIMPLE WORKFLOWS:**
- Use execute_workflow tool ONCE with the complete workflow structure
- Include all necessary steps in a well-structured workflow
- Focus on efficiency and completeness

**FOR COMPLEX WORKFLOWS:**
- Break down into logical sections and use execute_workflow tool for each section
- Iterative reflection: review previous sections and update if needed using input_schema with "section_update" property
- Continue until all sections are complete, then respond with COMPLETION_SIGNAL

**COMPLETION_SIGNAL:**
When the workflow is complete (whether simple or complex), respond with text only (no TOOL_USE) containing ONLY this JSON format:
{"name": "Workflow name", "description": "Brief description of what the workflow accomplishes"}
No need to re-generate the combined workflow that includes all sections, that will be handled by the workflow execution engine.

**WORKFLOW DESIGN PRINCIPLES:**
- **Keep workflows focused and minimal** - only include steps that directly address the core logic in WORKFLOW_DESCRIPTION
- **Avoid unnecessary complexity** - don't add extra tool_call or user_input unless specifically required
- **Create meaningful conditions** - never use static comparisons that always evaluate the same way
- **Cautious on branch and loop** - Only use branches and loops when there are genuine decision points
- **Avoid nested containers** - use "sequence" only with multiple steps and "parallel" with multiple branches
- **Tool_call must be from AVAILABLE_TOOLS list** - Never create tool_call nodes with tools not explicitly provided
- **Event-based synchronization** - use wait_for_event when needed to synchronize workflow steps or receive notifications
"""

    # Initialize conversation
    messages = [
        {
            "role": "user",
            "content": f"""
Analyze this WORKFLOW_DESCRIPTION and create a structured workflow definition. Choose the most appropriate planning approach based on the complexity.

WORKFLOW_DESCRIPTION:
{workflow_description}

AVAILABLE_TOOLS:
{json.dumps(available_tools, indent=2)}

Because of {max_tokens} max_tokens per response limitation, please analyze the overall complexity and break down the plan into multiple logical sections if needed.

Start your analysis, then generate workflow section 1 via execute_workflow tool.
""",
        }
    ]

    # Execute the generative planning process using the core method
    workflow_plan = planner.iterative_planning(
        messages=messages,
        system_prompt=system_prompt,
        workflow_execution_tool=workflow_execution_tool,
        available_tools=available_tools,
    )
    return workflow_plan, planner.claude_messages


def reflect_plan(
    existing_workflow_plan,
    user_feedback,
    available_tools,
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    max_interactions=20,
    max_tokens=8000
):
    """
    Reflect on and update an existing workflow plan based on user feedback using continuous Claude model interactions.

    This utility function creates a IterativePlanner instance and uses its iterative_planning
    method to modify an existing workflow plan based on user feedback.

    Args:
        existing_workflow_plan (dict): The current workflow plan to be modified
        user_feedback (str): User feedback describing what changes are needed
        available_tools (list): List of available tools
        model_id (str): The Bedrock model ID to use for planning

    Returns:
        dict: The updated structured workflow plan as a dictionary following the workflow schema,
              ready for execution by the workflow execution engine
    """
    print(
        f"\n{Fore.BLUE}Starting continuous workflow reflection process...{Style.RESET_ALL}"
    )

    # Create planner core instance
    planner = IterativePlanner(model_id=model_id, max_interactions=max_interactions, max_tokens=max_tokens)

    # Define the workflow execution tool - same as generate_plan
    workflow_execution_tool = {
        "name": "execute_workflow",
        "description": "Execute a partial workflow plan section or update an existing section. This tool takes a structured workflow definition (following the workflow schema) as input for a specific section of the overall workflow. To update an existing section, include a 'section_update' property in the root level with the section number to update.",
        "input_schema": planner.workflow_schema,
    }

    # Construct the system prompt for continuous reflection with iterative updates
    system_prompt = """
You are an AI assistant specialized in workflow orchestration and planning. You will work with me to analyze an EXISTING_WORKFLOW_PLAN and USER_FEEDBACK, then generate a new workflow plan that incorporates the requested feedback.

**ADAPTIVE PLANNING STRATEGY:**
First, analyze the EXISTING_WORKFLOW_PLAN and USER_FEEDBACK to determine complexity:
- **Simple workflows** (few steps, straightforward logic): Generate the complete workflow in ONE interaction
- **Complex workflows** (many steps, complex logic, multiple branches): Use iterative planning with multiple sections

**TOOL_USE STOP:**
- To generate new section, use execute_workflow tool with "name", "descripion", "root" properties
- To update previous section, use execute_workflow tool with "name", "descripion", "root" and "section_update" properties
- Always include "root" property with the structured workflow definition in input_schema

**FOR SIMPLE WORKFLOWS:**
- Use execute_workflow tool ONCE with the complete workflow structure
- Include all necessary steps in a well-structured workflow
- Focus on efficiency and completeness

**FOR COMPLEX WORKFLOWS:**
- Break down into logical sections and use execute_workflow tool for each section
- Iterative reflection: review previous sections and update if needed using input_schema with "section_update" property
- Continue until all sections are complete, then respond with COMPLETION_SIGNAL

**COMPLETION_SIGNAL:**
When the workflow is complete (whether simple or complex), respond with text only (no TOOL_USE) containing ONLY this JSON format:
{"name": "Workflow name", "description": "Brief description of what the workflow accomplishes"}
No need to re-generate the combined workflow that includes all sections, that will be handled by the workflow execution engine.

**WORKFLOW DESIGN PRINCIPLES:**
- **Focus only on changes requested** in the user feedback - do not add unrequested features
- **Preserve existing workflow elements** that are not mentioned in the feedback
- **Avoid unnecessary complexity** - don't add extra tool_call or user_input unless specifically required
- **Tool_call must be from AVAILABLE_TOOLS list** - Never create tool_call nodes with tools not explicitly provided
- **Event-based synchronization** - use wait_for_event when needed to synchronize workflow steps or receive notifications
"""

    # Initialize conversation with the existing workflow and feedback
    messages = [
        {
            "role": "user",
            "content": f"""
I need you to help me generate a new workflow plan based on EXISTING_WORKFLOW_PLAN, specific USER_FEEDBACK and AVAILABLE_TOOLS.

EXISTING_WORKFLOW_PLAN:
{json.dumps(existing_workflow_plan, indent=2)}

USER_FEEDBACK:
{user_feedback}

AVAILABLE_TOOLS:
{json.dumps(available_tools, indent=2)}

Because of {max_tokens} max_tokens per response limitation, please analyze the overall complexity and break down the plan into multiple logical sections if needed.

Start your analysis, then generate workflow section 1 via execute_workflow tool.
""",
        }
    ]

    # Execute the reflection process using the core method
    updated_plan = planner.iterative_planning(
        messages=messages,
        system_prompt=system_prompt,
        workflow_execution_tool=workflow_execution_tool,
        available_tools=available_tools,
    )
    return updated_plan, planner.claude_messages
