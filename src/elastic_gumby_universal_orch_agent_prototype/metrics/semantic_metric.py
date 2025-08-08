"""
Semantic Metric for Workflow Schema Comparison

This module provides functions to calculate semantic accuracy between workflows
defined in the workflow_schema.json format. It focuses on variable usage patterns
and tool call positioning to evaluate semantic similarity.
"""

import re

class SemanticMetric:
    """
    A class for calculating semantic metrics between workflows.
    
    This class provides methods to analyze variable usage patterns, tool call positioning,
    and overall semantic similarity between workflows.
    """
    
    def __init__(self):
        """Initialize the SemanticMetric analyzer."""
        pass
    
    def workflow_semantic_analysis(self, generated_workflow, reference_workflow):
        """
        Perform comprehensive semantic analysis between generated and reference workflows.
        
        Args:
            generated_workflow: The generated workflow to analyze
            reference_workflow: The reference workflow to compare against
            
        Returns:
            dict: Analysis results including tool call and data flow similarities
        """
        tool_call_result = self.tool_call_analysis(generated_workflow, reference_workflow)
        data_flow_result = self.data_flow_analysis(generated_workflow, reference_workflow)

        # Return detailed analysis with reduced variable emphasis
        return {
            "tool_call_similarity": tool_call_result["tool_call_similarity"],
            "average_tool_call_similarity": tool_call_result["average_tool_call_similarity"],
            "missing_tools": tool_call_result["missing_tools"],
            
            "missing_variables": data_flow_result["missing_variables"],
            "average_variable_definition_similarity": data_flow_result["average_variable_definition_similarity"],
            "variable_usage_similarity": data_flow_result["variable_usage_similarity"],
            "average_variable_usage_similarity": data_flow_result["average_variable_usage_similarity"]
        }
        

    def tool_call_analysis(self, generated_workflow, reference_workflow):
        """
        Analyze tool call similarities between workflows.
        
        Args:
            generated_workflow: The generated workflow to analyze
            reference_workflow: The reference workflow to compare against
            
        Returns:
            dict: Tool call analysis results
        """
        # Extract tool calls
        gen_tools = self._extract_tool_calls(generated_workflow)
        ref_tools = self._extract_tool_calls(reference_workflow)

        if not ref_tools:
            # Return consistent dictionary structure even when no reference tools
            return {
                "tool_call_similarity": {},
                "average_tool_call_similarity": 1.0 if not gen_tools else 0.0,
                "missing_tools": []
            }
        
        # Calculate tool coverage and similarity
        tool_call_similarity = {}
        missing_tools = []
        average_tool_call_similarity = 0.0
        for tool_name, ref_calls in ref_tools.items():
            tool_call_similarity[tool_name] = 0.0
            if tool_name not in gen_tools:
                missing_tools.append(tool_name)
                continue
            
            # For each reference call, find best matching call in generated workflow
            for ref_call in ref_calls:
                best_match_score = 0
                for cand_call in gen_tools[tool_name]:
                    # Calculate path similarity
                    path_sim = self._calculate_tool_path_similarity(ref_call["path"], cand_call["path"])
                    # Calculate parameter similarity
                    param_sim = self._calculate_tool_parameter_similarity(ref_call["parameters"], cand_call["parameters"])
                    # Calculate overall match score for this pair
                    match_score = (param_sim + path_sim) / 2
                    best_match_score = max(best_match_score, match_score)

                tool_call_similarity[tool_name] += best_match_score
            # Normalize by number of reference calls
            tool_call_similarity[tool_name] /= len(ref_calls)
            average_tool_call_similarity += tool_call_similarity[tool_name]
        
        average_tool_call_similarity /= len(ref_tools)

        return {
            "tool_call_similarity": tool_call_similarity,
            "average_tool_call_similarity": average_tool_call_similarity,
            "missing_tools": missing_tools
        }
    
    def _extract_tool_calls(self, workflow):
        """
        Extract tool calls from a workflow structure.
        
        Args:
            workflow: The workflow to extract tool calls from
            
        Returns:
            dict: Dictionary of tool calls organized by tool name
        """
        tool_calls = {}

        def traverse(node, path: str):
            # Handle the root property if present
            if "root" in node and isinstance(node["root"], dict):
                return traverse(node["root"], path)

            node_type = node.get("type", "unknown")

            # Check if current node is a tool call
            if node_type == "tool_call":
                tool_name = node.get("toolName")
                if tool_name:
                    if tool_name not in tool_calls:
                        tool_calls[tool_name] = []
                    tool_calls[tool_name].append({
                        "path": path + ".tool_call",
                        "parameters": node.get("parameters", {}),
                    })

            # Traverse children based on node type
            if node_type == "sequence" and "steps" in node:
                for i, step in enumerate(node["steps"]):
                    traverse(step, path + f".sequence.{i}")
            elif node_type == "parallel" and "branches" in node:
                for i, branch in enumerate(node["branches"]):
                    traverse(branch, path + f".parallel.{i}")
            elif node_type == "branch":
                traverse(node["ifTrue"], path + ".branch.true")
                traverse(node["ifFalse"], path + ".branch.false")
            elif node_type == "loop" and "body" in node:
                traverse(node["body"], path + ".loop.body")
            elif node_type == "wait_for_event" and "onTimeout" in node:
                    traverse(node["onTimeout"], path + ".wait_for_event.onTimeout")

        # Start traversal
        traverse(workflow, "root")

        return tool_calls

    def _calculate_tool_path_similarity(self, ref_path: str, cand_path: str):

        ref_tokens = ref_path.split(".")
        cand_tokens = cand_path.split(".")

        def token_levenshtein(seq1, seq2):
            if len(seq1) < len(seq2):
                return token_levenshtein(seq2, seq1)
            if len(seq2) == 0:
                return len(seq1)

            previous_row = range(len(seq2) + 1)
            for i, tok1 in enumerate(seq1):
                current_row = [i + 1]
                for j, tok2 in enumerate(seq2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (tok1 != tok2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        dist = token_levenshtein(ref_tokens, cand_tokens)
        max_len = max(len(ref_tokens), len(cand_tokens))
        return 1 - (dist / max_len) if max_len > 0 else 1.0

    def _calculate_tool_parameter_similarity(self, ref_params, cand_params):
        """
        Calculate similarity between tool parameters.
        
        Args:
            ref_params: Reference parameters
            cand_params: Candidate parameters
            
        Returns:
            float: Parameter similarity score
        """
        if not ref_params:
            return 1.0 if not cand_params else 0.0

        # Get all parameter keys, might use different optional keys
        all_keys = set(ref_params.keys()).union(set(cand_params.keys()))

        if not all_keys:
            return 1.0

        matches = 0

        for key in all_keys:
            if key in ref_params and key in cand_params:
                ref_val = ref_params[key]
                cand_val = cand_params[key]

                # Compare parameter values
                if ref_val == cand_val:
                    matches += 1.0
                elif isinstance(ref_val, str) and isinstance(cand_val, str):
                    # Check for variable references
                    ref_vars = set(re.findall(r"\${([^}]+)}", ref_val))
                    cand_vars = set(re.findall(r"\${([^}]+)}", cand_val))

                    if ref_vars and cand_vars:
                        # If both use variables, compare the variable patterns
                        var_similarity = len(ref_vars.intersection(cand_vars)) / len(ref_vars.union(cand_vars))
                        matches += var_similarity
                    else:
                        matches += 0.5  # Different reference variables but at least this parameter exists
                elif type(ref_val) == type(cand_val):
                    # If both are of the same object type but different values
                    matches += 0.5 

        return matches / len(all_keys)
    
    def data_flow_analysis(self, generated_workflow, reference_workflow):
        """
        Analyze data flow patterns between workflows.
        
        Args:
            generated_workflow: The generated workflow to analyze
            reference_workflow: The reference workflow to compare against
            
        Returns:
            dict: Data flow analysis results
        """
        # Extract variable definitions and usages
        gen_defs, gen_usages = self._extract_variables(generated_workflow)
        ref_defs, ref_usages = self._extract_variables(reference_workflow)
        
        # Calculate variable definition similarity scores
        missed_vars = []
        if not ref_defs:
            avg_definition_similarity = 1.0 if not gen_defs else 0.0
        else:
            total_def_matches = 0
            for ref_var, ref_context in ref_defs.items():
                if ref_var in gen_defs:
                    if ref_context == gen_defs[ref_var]:
                        total_def_matches += 1.0
                    else:
                        total_def_matches += 0.5
                else:
                    # Variable defined in reference but not in generated workflow
                    missed_vars.append(f"{ref_context} -> {ref_var}")
            avg_definition_similarity = total_def_matches / len(ref_defs)

        # Calculate variable usage similarity scores, usage patterns for each variable
        vars_usage_similarity = {}
        if not ref_usages:
            avg_usage_similarity = 1.0 if not gen_usages else 0.0
        else:
            total_usage_context_similarity = 0
            for ref_var, ref_contexts in ref_usages.items():
                if ref_var in gen_usages:
                    gen_contexts = gen_usages[ref_var]
                    # Calculate Jaccard similarity for contexts
                    intersection = len(set(ref_contexts).intersection(set(gen_contexts)))
                    union = len(set(ref_contexts).union(set(gen_contexts)))
                    context_similarity = intersection / union if union > 0 else 0
                    vars_usage_similarity[ref_var] = context_similarity
                    total_usage_context_similarity += context_similarity
            avg_usage_similarity = total_usage_context_similarity / len(ref_usages)

        return {
            "missing_variables": missed_vars,
            "average_variable_definition_similarity": avg_definition_similarity,
            "variable_usage_similarity": vars_usage_similarity,
            "average_variable_usage_similarity": avg_usage_similarity
        }

    def _extract_variables(self, workflow):
        """
        Extract variable definitions and usages from a workflow.
        
        Args:
            workflow: The workflow to extract variables from
            
        Returns:
            tuple: (definitions dict, usages dict)
        """
        definitions = {}
        usages = {}

        def find_variables_in_string(s, context):
            # Find all ${variable} patterns
            var_pattern = r"\${([^}]+)}"
            matches = re.findall(var_pattern, s)
            for var in matches:
                if usages.get(var) is None:
                    usages[var] = []
                usages[var].append(context)

        def traverse_condition(condition):
            if condition.get("type") == "comparison":
                find_variables_in_string(condition.get("left", ""), "condition.left")
                find_variables_in_string(condition.get("right", ""), "condition.right")
            elif condition.get("type") == "logical":
                for i, subcond in enumerate(condition.get("conditions", [])):
                    traverse_condition(subcond)

        def traverse(node):
            # Handle the root property if present
            if "root" in node and isinstance(node["root"], dict):
                return traverse(node["root"])

            node_type = node.get("type", "unknown")
            # Check for outputVariable for variable definitions
            if "outputVariable" in node:
                if node_type == "wait_for_event":
                    context = f"wait_for_event.{node.get('eventSource')}"
                elif node_type == "tool_call":
                    context = f"tool_call.{node.get('toolName')}"
                elif node_type == "user_input":
                    context = f"user_input"
                # each variable can only be defined once
                definitions[node["outputVariable"]] = context
            
            # Check all string properties for variable usages
            if node_type == "wait_for_event":
                find_variables_in_string(node.get("entityId", ""), "wait_for_event.entityId")
            elif node_type == "user_input":
                find_variables_in_string(node.get("prompt", ""), "user_input.prompt")
            elif node_type == "tool_call":
                for key, value in node.get("parameters", {}).items():
                    find_variables_in_string(str(value), f"tool_call.{node.get('toolName')}.{key}")
            elif node_type == "branch":
                traverse_condition(node.get("condition", {}))
            elif node_type == "loop":
                traverse_condition(node.get("condition", {}))
            
            # Traverse children based on node type
            if node_type == "sequence" and "steps" in node:
                for step in node["steps"]:
                    traverse(step)
            elif node_type == "parallel" and "branches" in node:
                for branch in node["branches"]:
                    traverse(branch)
            elif node_type == "branch":
                traverse(node["ifTrue"])
                traverse(node["ifFalse"])
            elif node_type == "loop":
                traverse(node["body"])
            elif node_type == "wait_for_event" and "onTimeout" in node:
                traverse(node["onTimeout"])

        # Start traversal
        traverse(workflow)

        return definitions, usages
