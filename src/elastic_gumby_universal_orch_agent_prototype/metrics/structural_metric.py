"""
Structural Metric for Workflow Schema Comparison

This module provides functions to calculate structural accuracy between workflows
defined in the workflow_schema.json format. It implements weighted node types and
hierarchical matching to provide a more nuanced comparison.
"""

class StructuralMetric:
    """
    A class for calculating structural metrics between workflows.
    
    This class provides methods to analyze workflow structure, node types,
    and hierarchical patterns between workflows.
    """
    
    def __init__(self):
        """Initialize the StructuralMetric analyzer."""
        pass

    def workflow_structural_analysis(self, generated_workflow, reference_workflow):
        """
        Provide detailed analysis of workflow comparison with enhanced structural insights.

        Args:
            generated_workflow (dict): The generated workflow from planner 
            reference_workflow (dict): The reference workflow to compare against

        Returns:
            dict: Detailed analysis of the comparison including score, matches, misses, and structural insights
        """
        # Extract subtrees
        generated_tree = self._build_workflow_tree(generated_workflow)
        reference_tree = self._build_workflow_tree(reference_workflow)
        generated_subtrees = self._get_subtrees(generated_tree)
        reference_subtrees = self._get_subtrees(reference_tree)

        # Track matches and misses with detailed information
        matches = []
        misses = []

        # Analyze each reference subtree
        total_reference_weight = 0
        total_matched_weight = 0
        for ref_subtree in reference_subtrees:
            matched = False
            total_reference_weight += ref_subtree["tree_weight"]
            # Look for matching subtree in candidate
            for generated_subtree in generated_subtrees:
                if self._subtrees_match(ref_subtree, generated_subtree):
                    matches.append({
                        "type": ref_subtree["type"],
                        "depth": ref_subtree["depth"],
                        "weight": ref_subtree["tree_weight"],
                        "children_count": len(ref_subtree.get("children", [])),
                        "matched_depth": generated_subtree["depth"],
                        "depth_difference": abs(ref_subtree["depth"] - generated_subtree["depth"]),
                    })
                    matched = True
                    total_matched_weight += ref_subtree["tree_weight"]
                    break

            if not matched:
                misses.append({
                    "type": ref_subtree["type"],
                    "depth": ref_subtree["depth"],
                    "weight": ref_subtree["tree_weight"],
                    "children_count": len(ref_subtree.get("children", [])),
                    "children_types": [child["type"] for child in ref_subtree.get("children", [])],
                })

        # Calculate subtree match ratio
        subtree_match_ratio = len(matches) / len(reference_subtrees) if reference_subtrees else 1.0
        # Calculate weighted subtree match accuracy
        weighted_subtree_match_accuracy = total_matched_weight / total_reference_weight if total_reference_weight > 0 else 0

        return {
            "subtree_match_ratio": subtree_match_ratio,
            "weighted_subtree_match_accuracy": weighted_subtree_match_accuracy,
            "matches": matches,
            "misses": misses,
            "match_count": len(matches),
            "miss_count": len(misses),
            "total_subtrees": len(reference_subtrees),
            "detailed_breakdown": {
                "action_nodes_matched": len(
                    [m for m in matches if m["type"] in ["tool_call", "user_input", "branch", "loop", "wait_for_event"]]
                ),
                "container_nodes_matched": len(
                    [m for m in matches if m["type"] in ["sequence", "parallel"]]
                ),
                "action_nodes_missing": len(
                    [m for m in misses if m["type"] in ["tool_call", "user_input", "branch", "loop", "wait_for_event"]]
                ),
                "container_nodes_missing": len(
                    [m for m in misses if m["type"] in ["sequence", "parallel"]]
                ),
                "deep_structures_matched": len([m for m in matches if m["depth"] >= 2]),
                "shallow_structures_matched": len([m for m in matches if m["depth"] < 2]),
            },
        }


    def _build_workflow_tree(self, workflow, depth=0):
        """
        Extract a tree representation from a workflow with depth information.

        Args:
            workflow (dict): The workflow to extract a tree from
            depth (int): The current depth in the tree (used recursively)

        Returns:
            dict: A tree representation with type, children, and depth information
        """
        # Handle the root property if present
        if "root" in workflow:
            return self._build_workflow_tree(workflow["root"], depth)

        # All structures should have explicit types
        node_type = workflow.get("type", "unknown")
        node_weight = 2.0 if node_type in ["tool_call", "user_input", "branch", "loop", "wait_for_event"] else 1.0
        depth_weight = 1.0 / (depth + 1)
        tree = {"type": node_type, "children": [], "depth": depth, "tree_weight": node_weight * depth_weight}

        # Handle different node types
        if node_type == "sequence" and "steps" in workflow:
            for step in workflow["steps"]:
                tree["children"].append(self._build_workflow_tree(step, depth + 1))
        elif node_type == "parallel" and "branches" in workflow:
            for branch in workflow["branches"]:
                tree["children"].append(self._build_workflow_tree(branch, depth + 1))
        elif node_type == "branch":
            tree["children"].append(self._build_workflow_tree(workflow["ifTrue"], depth + 1))
            tree["children"].append(self._build_workflow_tree(workflow["ifFalse"], depth + 1))
        elif node_type == "loop" and "body" in workflow:
            tree["children"].append(self._build_workflow_tree(workflow["body"], depth + 1))
        elif node_type == "wait_for_event" and "onTimeout" in workflow:
            tree["children"].append(self._build_workflow_tree(workflow["onTimeout"], depth + 1))

        return tree

    def _get_subtrees(self, tree):
        """
        """
        subtrees = [tree]
        for child in tree.get("children", []):
            subtrees.extend(self._get_subtrees(child))
        return subtrees


    def _subtrees_match(self, tree1, tree2):
        """
        Check if two subtrees match structurally.

        Args:
            tree1 (dict): First subtree
            tree2 (dict): Second subtree

        Returns:
            bool: True if the subtrees match structurally, False otherwise
        """
        if tree1["type"] != tree2["type"]:
            return False

        if len(tree1.get("children", [])) != len(tree2.get("children", [])):
            return False

        # For leaf nodes, just match the type
        if not tree1.get("children") and not tree2.get("children"):
            return True

        # For non-leaf nodes, check if children types match (order matters)
        child_types1 = [child["type"] for child in tree1.get("children", [])]
        child_types2 = [child["type"] for child in tree2.get("children", [])]

        return child_types1 == child_types2
