"""
Workflow to Amazon States Language (ASL) Transformer

This module provides functionality to transform workflow plans defined in the workflow schema (data_schema/workflow_schema.json)
into State Machine in Amazon States Language (data_schema/states_language_schema.json) that can be executed by AWS Step Function.

The transformer handles:
- Sequence Container (States with Next transitions)
- Parallel Container (Parallel state)
- Tool_call (Task state)
- User_input (Task state with Task Token service integration)
- Wait_for_event (Wait state + Task state with Task Token service integration + Choice state for timeout handling)
- Branch (Choice state with Default for false branch)
- Loop (Choice state + Loop body states + Pass state)
"""

import json
from colorama import Fore, Style
import re
from typing import Dict, Any, Tuple
from pathlib import Path
import boto3


class StateMachineTransformer:
    """
    Transforms workflow plans into Amazon States Language state machines.
    
    This class handles the conversion from the flexible workflow schema format
    to the structured ASL format required by AWS Step Functions, with support
    for JSONata expressions and comprehensive error handling.
    
    Attributes:
        state_counter (Dict[str, int]): Counter for generating unique state names
        available_tools_resource (Dict[str, str]): Mapping of tool names to their AWS resource ARNs
        state_variables (Dict[str, Any]): State machine variables and their value range for demo purposes
        state_machine (Dict[str, Any]): The generated ASL state machine definition
    """
    
    def __init__(self, available_tools: list[Dict[str, Any]]):
        """
        Initialize the StateMachineTransformer with available tools.
        
        Args:
            available_tools: List of dictionaries containing tool definitions with
                           name and resource fields
        """
        self.state_counter = {}
        self.available_tools_resource = {}
        for tool in available_tools:
            self.available_tools_resource[tool["name"]] = tool["resource"]
        self.state_variables = {}  # Store state machine variables and their demo values
        self.state_machine = {}  # The final ASL state machine definition
        self.var_pattern = r'\{\%\s*\$([a-zA-Z_][\w\.]*)\s*\%\}' # Pattern to match "{% $variableName.property %}"

    def _get_state_name(self, state_type: str) -> str:
        """
        Generate a unique state name for the given state type.
        
        This method maintains internal counters for each state type to ensure
        all state names are unique within the state machine. State names follow
        the pattern: {state_type}_{counter}
        
        Args:
            state_type: The type of state (e.g., 'UserInput', 'Parallel', tool name)
            
        Returns:
            A unique state name string in the format "{state_type}_{counter}"
        """
        if state_type not in self.state_counter:
            self.state_counter[state_type] = 1
        else:
            self.state_counter[state_type] += 1
        return f"{state_type}_{self.state_counter[state_type]}"
    
    def save_state_machine(self, workflow_plan: Dict[str, Any], save_dir: 'Path') -> None:
        """
        Save the generated state machine and execution input to files.
        
        This method transforms the workflow plan into a state machine and saves both
        the ASL definition and execution input variables to separate JSON files.
        
        Args:
            workflow_plan: Dictionary containing the workflow plan definition
            save_dir: Directory path where the output files will be saved
            
        Returns:
            None
        """
        self.transform_workflow(workflow_plan)
        SFN_client = boto3.client('stepfunctions')
        try:
            response = SFN_client.validate_state_machine_definition(definition=json.dumps(self.state_machine))
            validation_result = response.get('result')
            if validation_result == "OK":
                print(f"{Fore.GREEN}✅ State Machine definition is valid{Style.RESET_ALL}")
                for diag in response.get('diagnostics', []):
                    print(f"{Fore.YELLOW}⚠️  {diag['severity']}: {diag['code']}, {diag['message']} at {diag['location']}{Style.RESET_ALL}")
            else:  # Should be "FAIL"
                print(f"{Fore.RED}State Machine definition is invalid: {validation_result}{Style.RESET_ALL}")
                for diag in response.get('diagnostics', []):
                    print(f"{Fore.RED}❌ {diag['severity']}: {diag['code']}, {diag['message']} at {diag['location']}{Style.RESET_ALL}")
                return
        except Exception as e:
            print(f"{Fore.RED}❌ Validate State Machine Definition Process Failed: {e}{Style.RESET_ALL}")
            return
        
        asl_output_path = save_dir / "state_machine.asl.json"
        exec_input_path = save_dir / "exec_input.json"
        with open(asl_output_path, "w") as f:
            json.dump(self.state_machine, f, indent=2)
        with open(exec_input_path, "w") as f:
            json.dump(self.state_variables, f, indent=2)
        print(f"{Fore.GREEN}✅ State Machine saved to {asl_output_path}{Style.RESET_ALL}") 
        print(f"{Fore.GREEN}✅ Execution Input saved to {exec_input_path}{Style.RESET_ALL}")
    
    def transform_workflow(self, workflow_plan: Dict[str, Any]) -> None:
        """
        Transform a complete workflow plan into an ASL state machine.
        
        This method processes the workflow plan, extracts metadata, transforms the root element,
        adds demo return values for testing, and builds the complete ASL state machine with
        proper initialization of state variables.
        
        Args:
            workflow_plan: Dictionary containing the workflow plan definition with metadata and root element
            
        Returns:
            Dictionary representing the complete ASL state machine definition
        """ 
        # Extract workflow metadata
        workflow_name = workflow_plan.get("name", "UnnamedWorkflow")
        workflow_description = workflow_plan.get("description", "")
        root_element = workflow_plan["root"]
        
        # Transform the root element
        states, start_state, end_states, assigned_vars = self._transform_container_or_node(root_element)
        
        returned_vars = set()
        # Add ReturnValueRange to Task states and initialize Pass states for Choice Variables
        for state_def in states.values():
            if state_def["Type"] == "Task" and "Assign" in state_def:
                # Add a demo return value for Task states
                returned_var = list(state_def["Assign"].keys())[0]
                flat_vars = {k: v for k, v in self.state_variables.items() if k.startswith(returned_var)}
                # Could be a simple value range list OR a dictionary of <var: value range list> pairs for nested properties
                state_def["Arguments"]["ReturnValueRange"] = flat_vars
                state_def["Assign"] = {k: "{% $states.result." + k + " %}" for k in flat_vars.keys()}
                returned_vars.update(flat_vars.keys()) # dict_keys object
            elif state_def["Type"] == "Pass" and state_def.get("Comment") in ["Choice Variables", "Parallel Variables"]:
                vars = list(state_def["Assign"].keys())
                flat_vars = {}
                for var in vars:
                    flat_vars.update({var: None for k in self.state_variables.keys() if k.startswith(var)})
                state_def["Assign"] = flat_vars
                returned_vars.update(flat_vars.keys())  # Track assigned variables in Pass states
        
        # Remove all keys in returned_vars from self.state_variables
        self.state_variables = {k: v for k, v in self.state_variables.items() if k not in returned_vars}

        initial_state_name = "Input State Variables"
        initial_state = {
            "Type": "Pass",
            "Comment": "Initialize state machine variables",
            "Assign": {
                var_name: "{% $states.input." + var_name + " %}"
                for var_name in self.state_variables.keys()
            },
            "Next": start_state
        }
        states[initial_state_name] = initial_state

        # Build the complete ASL state machine
        self.state_machine = {
            "Comment": f"{workflow_name}: {workflow_description}",
            "StartAt": initial_state_name,
            "QueryLanguage": "JSONata",
            "States": states
        }
    
    def _transform_container_or_node(self, element: Dict[str, Any]) -> Tuple[Dict[str, Any], str, list[str], set[str]]:
        """
        Transform a container or node element into ASL states based on its type.
        
        This method serves as a dispatcher that routes the transformation process to the
        appropriate specialized method based on the element type (sequence, parallel, 
        tool_call, user_input, wait_for_event, branch, or loop).
        
        Args:
            element: The workflow element to transform
            
        Returns:
            Tuple containing:
            - Dictionary of generated ASL states
            - Name of the start state
            - List of names of end states
            - Set of variables assigned in this container
            
        Raises:
            ValueError: If the element type is unknown or unsupported
        """
        element_type = element.get("type")
        
        if element_type == "sequence":
            return self._transform_sequence(element)
        elif element_type == "parallel":
            return self._transform_parallel(element)
        elif element_type == "tool_call":
            return self._transform_tool_call(element)
        elif element_type == "user_input":
            return self._transform_user_input(element)
        elif element_type == "wait_for_event":
            return self._transform_wait_for_event(element)
        elif element_type == "branch":
            return self._transform_branch(element)
        elif element_type == "loop":
            return self._transform_loop(element)
        else:
            raise ValueError(f"Unknown element type: {element_type}")
    
    def _transform_sequence(self, sequence: Dict[str, Any]) -> Tuple[Dict[str, Any], str, list[str], set[str]]:
        """
        Transform a sequence container into sequential ASL states.
        
        This method processes a sequence of workflow steps, transforming each step and
        chaining them together by setting the "Next" field of each state to point to
        the start state of the following step. The last state in the sequence is marked
        with "End": True.
        
        Args:
            sequence: Dictionary containing the sequence definition with a "steps" list
            
        Returns:
            Tuple containing:
            - Dictionary of generated ASL states
            - Name of the start state (first state in the sequence)
            - List of names of end states (last states in the sequence)
            - Set of variables assigned in this sequence
        """
        steps = sequence["steps"]
        all_states = {}
        assigned_variables = set()  # Track variables assigned in this container
        # Transform each step and chain them together
        start_state = None
        end_states = []
        
        for step in steps:
            step_states, step_start, step_ends, assigned_vars = self._transform_container_or_node(step)
            
            all_states.update(step_states) # Merge step states
            assigned_variables.update(assigned_vars)  # Collect assigned variables
            
            if start_state is None:
                start_state = step_start
            
            # Link previous states to current state
            for state in end_states:
                del all_states[state]["End"]
                all_states[state]["Next"] = step_start
            
            # Update end states
            end_states = step_ends
        
        # Ensure the last state ends the sequence
        for state in end_states:
            all_states[state]["End"] = True
        
        return all_states, start_state, end_states, assigned_variables
    
    def _transform_parallel(self, parallel: Dict[str, Any]) -> Tuple[Dict[str, Any], str, list[str], set[str]]:
        """
        Transform a parallel container into a Parallel ASL state.
        
        This method creates an ASL Parallel state that executes multiple branches
        concurrently. Each branch is transformed into its own sub-state machine
        with its own start state and states dictionary.
        
        Args:
            parallel: Dictionary containing the parallel definition with a "branches" list
            
        Returns:
            Tuple containing:
            - Dictionary with the generated Parallel state
            - Name of the Parallel state
            - List containing the name of the Parallel state (as it is both start and end)
            - Set of variables assigned in this parallel container
        """
        parallel_state_name = self._get_state_name("Parallel")
        pass_state_name = self._get_state_name("Pass")
        branches = parallel["branches"]
        
        # Transform each branch
        branch_definitions = []
        all_states = {}
        assigned_variables = set()  # Track variables assigned in this parallel container
        
        for branch in branches:
            branch_states, branch_start, branch_ends, assigned_vars = self._transform_container_or_node(branch)
            
            # Create a sub-state machine for this branch
            branch_definition = {
                "StartAt": branch_start,
                "States": branch_states
            }
            branch_definitions.append(branch_definition)
            assigned_variables.update(assigned_vars)  # Collect assigned variables
        
        # Create the Parallel state
        parallel_state = {
            "Type": "Parallel",
            "Comment": parallel.get("description", "Parallel execution"),
            "Branches": branch_definitions,
            "Next": pass_state_name
        }
        pass_state = {
            "Type": "Pass",
            "Comment": "Parallel Variables",
            "Assign": {
                var_name: None  # Place holder, will be replaced by actual JSON object in transform_workflow
                for var_name in assigned_variables
            },
            "End": True 
        }
        
        all_states = {
            parallel_state_name: parallel_state,
            pass_state_name: pass_state
        }
        return all_states, parallel_state_name, [pass_state_name], assigned_variables
    
    def _transform_tool_call(self, tool_call: Dict[str, Any]) -> Tuple[Dict[str, Any], str, list[str], set[str]]:
        """
        Transform a tool call node into a JSONata Task ASL state.
        
        This method creates an ASL Task state that invokes an AWS resource (like Lambda)
        with the specified parameters. It handles output variable assignment and optional
        error handling through the Catch field if an errorHandler is specified.
        
        Args:
            tool_call: Dictionary containing the tool call definition with toolName,
                      parameters, and optional outputVariable and errorHandler
            
        Returns:
            Tuple containing:
            - Dictionary of generated ASL states (including error handler states if present)
            - Name of the Task state
            - List of names of end states (includes error handler end states if present)
            - Set of variables assigned in this tool call
        """
        state_name = self._get_state_name(f"{tool_call['toolName']}")
        
        # Build the JSONata task state
        task_state = {
            "QueryLanguage": "JSONata",  # Explicitly set for JSONata tasks
            "Type": "Task",
            "Resource": self.available_tools_resource[tool_call["toolName"]],
            "Comment": tool_call.get("description", f"Call {tool_call['toolName']}"),
            "Arguments": self._collect_parameters(tool_call.get("parameters", {})),  # Use Arguments for JSONata tasks
            "End": True
        }
        all_states = {state_name: task_state}
        assigned_variables = set()  # Track variables assigned in this tool call
        
        # Assign outputVariable to State Machine Variable
        # Reference: https://docs.aws.amazon.com/step-functions/latest/dg/workflow-variables.html
        if "outputVariable" in tool_call:
            var_name = tool_call["outputVariable"]
            task_state["Assign"] = {
                var_name: "{% $states.result %}"   # Assign variable with the API or sub-workflow's result (if successful)
            }
            assigned_variables.add(var_name)  # Track assigned variable
        
        # Handle error handling
        if "errorHandler" in tool_call:
            error_states, error_start, error_ends, assigned_vars = self._transform_container_or_node(tool_call["errorHandler"])
            
            task_state["Catch"] = [{
                "ErrorEquals": ["States.ALL"],
                "Next": error_start,
                "Comment": "Handle tool call errors"
            }]
            
            all_states.update(error_states)
            assigned_variables.update(assigned_vars)  # Collect assigned variables from error handler
            
            return all_states, state_name, error_ends.append(state_name), assigned_variables 
        
        return all_states, state_name, [state_name], assigned_variables
    
    def _transform_user_input(self, user_input: Dict[str, Any]) -> Tuple[Dict[str, Any], str, list[str], set[str]]:
        """
        Transform a user input node into a Task state with Task Token service integration.
        
        This method creates an ASL Task state that waits for human input using the
        Task Token service integration pattern. It builds arguments for the task including
        prompt and input type, and handles optional options and output variable assignment.
        
        Args:
            user_input: Dictionary containing the user input definition with prompt,
                       optional inputType, options, and outputVariable
            
        Returns:
            Tuple containing:
            - Dictionary with the generated Task state
            - Name of the Task state
            - List containing the name of the Task state (as it is both start and end)
            - Set of variables assigned in this user input
        """
        state_name = self._get_state_name(f"UserInput")
        
        # Build arguments for user input / human approval steps in the workflow via TaskState with a callback and a task token
        # Reference: https://docs.aws.amazon.com/step-functions/latest/dg/connect-to-resource.html#connect-wait-token
        arguments = {
            "prompt": user_input.get("prompt", "Prompt for user input"),
            "inputType": user_input.get("inputType", "Input Type")
        }
        
        if "options" in user_input:
            arguments["options"] = user_input["options"]
        
        # Create JSONata task state for user input
        task_state = {
            "Type": "Task",
            "QueryLanguage": "JSONata",  # Required for JSONata tasks
            "Resource": "arn:aws:lambda:us-west-2:411097365838:function:ElasticGumbyUniversalOrchAgentPrototype-DemoLambda",  # Custom resource for user input
            "Comment": "Wait for user input",
            "Arguments": arguments,
            "End": True  # Default transition
        }
        assigned_variables = set()
        
        # Assign outputVariable to State Machine Variable
        if "outputVariable" in user_input:
            var_name = user_input["outputVariable"]
            task_state["Assign"] = {
                var_name: "{% $states.result %}"  # Assign variable with the API or sub-workflow's result (if successful)
            }
            assigned_variables.add(var_name)  # Track assigned variable
        
        return {state_name: task_state}, state_name, [state_name], assigned_variables 
    
    def _transform_wait_for_event(self, wait_event: Dict[str, Any]) -> Tuple[Dict[str, Any], str, list[str], set[str]]:
        """
        Transform a wait for event node into a Wait state followed by a Task state with Task Token integration.
        
        This method creates a sequence of ASL states that:
        1. Waits for a specified duration (Wait state)
        2. Creates a Task state that waits for an external event using Task Token integration
        3. Optionally handles timeout scenarios with a Choice state if onTimeout is specified
        
        Args:
            wait_event: Dictionary containing the wait event definition with eventType,
                       eventSource, optional entityId, timeout, outputVariable, and onTimeout
            
        Returns:
            Tuple containing:
            - Dictionary of generated ASL states
            - Name of the Wait state (entry point)
            - List of names of end states
            - Set of variables assigned in this wait for event
        """
        wait_seconds = min(wait_event.get('timeout', 10), 10) # Default and set a maximum of 10 seconds for demonstration
        wait_state_name = self._get_state_name(f"Wait{wait_seconds}Seconds")
        wait_for_task_name = self._get_state_name(f"WaitFor_{wait_event['eventType']}")
        wait_state = {
            "Type": "Wait",
            "Seconds": wait_seconds, 
            "Next": wait_for_task_name,  # Transition to the next state after waiting
        }
        # Create a task state that waits for the event, using Task Token service integration pattern
        wait_for_task = {
            "Type": "Task",
            "Comment": f"Wait for {wait_event['eventType']} from {wait_event['eventSource']}",
            "Resource": "arn:aws:lambda:us-west-2:411097365838:function:ElasticGumbyUniversalOrchAgentPrototype-DemoLambda", 
            "Arguments": {
                "eventType": wait_event["eventType"],
                "eventSource": wait_event["eventSource"]
            },
            # Reference: https://docs.aws.amazon.com/step-functions/latest/dg/connect-to-resource.html#wait-token-hearbeat
            "HeartbeatSeconds": wait_seconds, 
            "End": True
        }

        if "entityId" in wait_event:
            # Convert entityId to JSONata expression if it is a variable reference
            wait_for_task["Arguments"]["entityId"] = self._collect_state_varibles(wait_event["entityId"])

        assigned_variables = set()  

        # Assign outputVariable to State Machine Variable
        if "outputVariable" in wait_event:
            var_name = wait_event["outputVariable"]
            wait_for_task["Assign"] = {
                var_name: "{% $states.result %}"   # Assign variable with the event result
            }
            assigned_variables.add(var_name)  # Track assigned variable
        
        all_states = {
            wait_state_name: wait_state,
            wait_for_task_name: wait_for_task
        }
        # Handle timeout handler via add another state
        if "onTimeout" in wait_event:
            wait_result_check_name = self._get_state_name(f"WaitFor_{wait_event['eventType']}_ResultCheck")
            pass_state_name = self._get_state_name("Pass")
            timeout_states, timeout_start, timeout_ends, assigned_vars = self._transform_container_or_node(wait_event["onTimeout"])
            assigned_variables.update(assigned_vars)  # Collect assigned variables from timeout handler
            wait_result_check_task = {
                "Type": "Choice",
                "Comment": f"Check result of {wait_event['eventType']}",
                "Choices": [{
                    "Condition": "{% 'error' in $states.input %}",
                    "Next": timeout_start  # Transition to result check state if successful
                }],
                "Default": pass_state_name,  # Transition back to wait state if no result
            }
            pass_task = {
                "Type": "Pass",
                "Comment": f"Received {wait_event['eventType']} in wait state",
                "End": True  # End the state machine after passing
            }
            # Remove End and add success transition
            del wait_for_task["End"]
            wait_for_task["Next"] = wait_result_check_name
            all_states.update(timeout_states)
            all_states[wait_result_check_name] = wait_result_check_task
            all_states[pass_state_name] = pass_task
            return all_states, wait_state_name, timeout_ends + [pass_state_name], assigned_variables
            
        return all_states, wait_state_name, [wait_for_task_name], assigned_variables
        

    def _transform_branch(self, branch: Dict[str, Any]) -> Tuple[Dict[str, Any], str, list[str], set[str]]:
        """
        Transform a branch node into a Choice ASL state.
        
        This method creates an ASL Choice state that evaluates a condition and directs
        the workflow to one of two branches based on the result. It transforms both the
        true and false branches and connects them to the Choice state.
        
        Args:
            branch: Dictionary containing the branch definition with condition,
                  ifTrue, and ifFalse branches
            
        Returns:
            Tuple containing:
            - Dictionary of generated ASL states (including both branches)
            - Name of the Choice state
            - List of names of end states from both branches
            - Set of variables assigned in this branch
        """
        choice_condition = self._convert_condition(branch["condition"])
        # Ensure state name length less than or equal to 80 Unicode characters
        condition_state_name = choice_condition if len(choice_condition) <= 70 else choice_condition[:70] + "..."
        condition_state_name = self._get_state_name(condition_state_name)  # Ensure unique state name
        
        # Transform the true and false branches
        true_states, true_start, true_ends, assigned_vars_true = self._transform_container_or_node(branch["ifTrue"])
        false_states, false_start, false_ends, assigned_vars_false = self._transform_container_or_node(branch["ifFalse"])
        
        # Create the Choice state
        choice_state = {
            "Type": "Choice",
            "Comment": branch.get("description", "Conditional branch"),
            "Choices": [{
                "Condition": choice_condition,
                "Next": true_start
            }],
            "Default": false_start
        }

        true_pass_state_name = self._get_state_name("Pass")
        false_pass_state_name = self._get_state_name("Pass")
        for state in true_ends:
            del true_states[state]["End"]
            true_states[state]["Next"] = true_pass_state_name
        for state in false_ends:    
            del false_states[state]["End"]
            false_states[state]["Next"] = false_pass_state_name
        true_pass = {
            "Type": "Pass",
            "Comment": "Choice Variables",
            "Assign": {
                var_name: None  # Place holder, will be replaced by actual JSON object in transform_workflow
                for var_name in assigned_vars_false
            },
            "End": True
        }
        false_pass = {
            "Type": "Pass",
            "Comment": "Choice Variables",
            "Assign": {
                var_name: None
                for var_name in assigned_vars_true
            },
            "End": True
        }
        
        # Combine all states
        all_states = {condition_state_name: choice_state}
        all_states.update(true_states)
        all_states.update(false_states)
        all_states[true_pass_state_name] = true_pass
        all_states[false_pass_state_name] = false_pass
        
        return all_states, condition_state_name, [true_pass_state_name, false_pass_state_name], set()
    
    def _transform_loop(self, loop: Dict[str, Any]) -> Tuple[Dict[str, Any], str, list[str], set[str]]:
        """
        Transform a loop node into recursive ASL states.
        
        This method creates a set of ASL states that implement a loop structure:
        1. A Choice state that evaluates the loop condition
        2. The loop body states that are executed when the condition is true
        3. A Pass state that serves as the exit point when the condition is false
        4. Connections from the end of the loop body back to the condition check
        
        Args:
            loop: Dictionary containing the loop definition with condition and body
            
        Returns:
            Tuple containing:
            - Dictionary of generated ASL states
            - Name of the condition Choice state (entry point)
            - List containing the name of the Pass state (exit point)
            - Set of variables assigned in this loop
        """
        # Create states for loop condition check and body
        loop_condition = self._convert_condition(loop["condition"])
        # Ensure state name length less than or equal to 80 Unicode characters
        condition_state_name = loop_condition if len(loop_condition) <= 70 else loop_condition[:70] + "..."
        condition_state_name = self._get_state_name(condition_state_name)  # Ensure unique state name
        body_states, body_start, body_ends, assigned_vars = self._transform_container_or_node(loop["body"])
        loop_pass_state_name = self._get_state_name("Pass")
        
        # Create condition check state
        condition_state = {
            "Type": "Choice",
            "Comment": "Check loop condition",
            "Choices": [{
                "Condition": loop_condition,
                "Next": body_start
            }],
            "Default": loop_pass_state_name
        }
        
        # Create end and iterator control states
        pass_state = {
            "Type": "Pass",
            "Comment": "Loop completed",
            "End": True
        }

        # Modify body to loop back to condition
        for state in body_ends:
            del body_states[state]["End"]
            body_states[state]["Next"] = condition_state_name
        
        # Combine all states
        all_states = {
            condition_state_name: condition_state,
            loop_pass_state_name: pass_state
        }
        all_states.update(body_states)

        if loop["condition"]["operator"] not in ["==", "!=", "in"]: # Must have used state variable as loop iterator in condition["left"]
            match = re.fullmatch(self.var_pattern, loop["condition"]["left"].strip())
            iterator = match.group(1).replace('.', '_') # Flatten variable name for Assign
            iterator_state_name = self._get_state_name("IteratorControl")
            iterator_state = {
                "Type": "Pass",
                "Comment": "Loop iterator increment",
                "Assign": {
                    iterator: "{% $" + iterator + " + 1 %}"  # Increment loop iterator
                },
                "Next": condition_state_name  # Loop back to condition check
            }
            for state in body_ends:
                body_states[state]["Next"] = iterator_state_name
            all_states[iterator_state_name] = iterator_state
        
        return all_states, condition_state_name, [loop_pass_state_name], assigned_vars
    
    def _convert_condition(self, condition: Dict[str, Any]) -> str:
        """
        Convert a workflow condition to a JSONata expression for use in Choice states.
        This method processes a condition dictionary, which can be either a comparison
        or a logical condition, and converts it to a JSONata expression that can be used
        in an ASL Choice state.
        
        Args:
            condition: Dictionary containing the condition definition with type, operator,
                          left, right, and optional conditions for logical operators
        Returns:    
            A JSONata expression string representing the condition, formatted for use in ASL Choice states
        Raises:
            ValueError: If the condition type is unknown or unsupported
        """
        condition_type = condition.get("type")
        
        if condition_type == "comparison":
            # Map operators to JSONata comparison operator, https://docs.jsonata.org/comparison-operators
            op = condition["operator"]
            right = condition["right"]
            if isinstance(right, str) and re.fullmatch(self.var_pattern, condition["right"].strip()):
                jsonata_left = self._collect_state_varibles(condition['left'], [1, 2, 3, 4, 5])
                jsonata_right = self._collect_state_varibles(condition["right"], [1, 2, 3, 4, 5])
                return jsonata_left[:-2] + " " + op + " " + jsonata_right[2:]
            elif isinstance(right, str) or right is None:
                # Right side must be a string literal for JSONata
                jsonata_right = str(right)
                jsonata_left = self._collect_state_varibles(condition['left'], [jsonata_right, f"NOT_{jsonata_right}"])
                return jsonata_left[:-2] + " " + op + " '" + jsonata_right + "' %}" #  string literals must be in single quotes (') in JSONata
            elif isinstance(right, bool): # bool is a subclass of int! 
                # Boolean comparison, and both sides are variable references
                jsonata_left = self._collect_state_varibles(condition['left'], [not right, right])
                return jsonata_left[:-2] + " " + op + " " + str(right).lower() + " %}" # convert to true/false in JSONata
            elif isinstance(right, (int, float)):
                # Numeric comparison, and both sides are variable references
                jsonata_left = self._collect_state_varibles(condition['left'], [right - 1, right, right + 1])
                return  jsonata_left[:-2] + " " + op + " " + str(right) + " %}"
            else:
                raise ValueError(f"Unsupported right value type: {type(right)} in condition {condition}")
        
        elif condition_type == "logical":
            operator = condition["operator"]
            conditions = condition["conditions"]
            
            if operator == "and":
                # Handle each sub-condition separately to avoid KeyError
                sub_conditions = []
                for c in conditions:
                    sub_condition = self._convert_condition(c)
                    sub_conditions.append(sub_condition)
                return "{% " + " and ".join(sub_condition[2:-2] for sub_condition in sub_conditions) + " %}"
            elif operator == "or":
                # Handle each sub-condition separately to avoid KeyError
                sub_conditions = []
                for c in conditions:
                    sub_condition = self._convert_condition(c)
                    sub_conditions.append(sub_condition)
                return "{% " + " or ".join(sub_condition[2:-2] for sub_condition in sub_conditions) + " %}"
        
        # This could never happen in a valid workflow, but handle gracefully
        raise ValueError(f"Unknown condition type: {condition_type}")
    
    def _collect_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert workflow parameters to JSONata Arguments with proper expression handling.
        
        This method recursively processes a parameters dictionary, converting string
        values that contain variable references to JSONata expressions and handling
        nested dictionaries appropriately.
        
        Args:
            parameters: Dictionary of parameter names and values to convert
            
        Returns:
            Dictionary with the same structure but with string values containing
            variable references converted to JSONata expressions
        """
        converted = {}
        
        for key, value in parameters.items():
            if isinstance(value, str):
                converted[key] = self._collect_state_varibles(value)
            elif isinstance(value, dict):
                converted[key] = self._collect_parameters(value)
            else:
                # Static values (numbers, booleans, etc.)
                converted[key] = value
        
        return converted
    
    def _collect_state_varibles(self, text: str, value_range: list = None) -> str:
        """
        Collect state variables from a string and convert them to JSONata expressions.
        This method searches for variable references in the form of {% $variableName %} 
        within the provided text and replaces them with JSONata expressions, 
        ensuring that the variable names are flattened to avoid dot notation in JSONata.
        Args:
            text: The input string containing variable references
            value_range: Optional list of values to assign to the variable, used for
                         initializing state variables in the state machine
        Returns:
            The input string with variable references replaced by JSONata expressions,
            and state variables initialized in self.state_variables if they are not already set.
        """        
        # Check if the entire string to match the pattern
        outer_pattern = r'^\{\%.*\%\}$'
        if re.fullmatch(outer_pattern, text.strip()):
            # Find all $variableName occurrences inside the block
            inner_pattern = r'\$([a-zA-Z_][\w\.]*)'
            variables = re.findall(inner_pattern, text)
            for var in variables:
                # Flatten variable name for Step Functions compatibility
                flat_var = var.replace('.', '_')
                if flat_var not in self.state_variables or self.state_variables[flat_var] is None:
                    self.state_variables[flat_var] = value_range
            
            def repl(match):
                return match.group(0).replace('.', '_')
            return re.sub(inner_pattern, repl, text)  # Return the falt variable name to avoid dot notation in JSONata
        else:
            return text