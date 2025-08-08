"""
Workflow Comparator

This module combines structural and semantic metrics to provide a comprehensive
evaluation of workflows defined in the workflow_schema.json format.
"""

from .semantic_metric import SemanticMetric
from .structural_metric import StructuralMetric

# Color codes for enhanced readability
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Header colors
    HEADER = "\033[1;36m"  # Bold Cyan
    SUBHEADER = "\033[1;33m"  # Bold Yellow
    SECTION = "\033[1;35m"  # Bold Magenta

    # Status colors
    SUCCESS = "\033[92m"  # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"  # Red
    INFO = "\033[94m"  # Blue

    # Accent colors
    SCORE = "\033[96m"  # Cyan
    METRIC = "\033[95m"  # Magenta
    VALUE = "\033[97m"  # White

def _colorize(text, color):
        """Apply color to text."""
        return f"{color}{text}{Colors.RESET}"

def format_score(score, threshold_good=0.8, threshold_ok=0.6):
    """Format score with color based on value."""
    if score >= threshold_good:
        return _colorize(f"{score:.3f}", Colors.SUCCESS)
    elif score >= threshold_ok:
        return _colorize(f"{score:.3f}", Colors.WARNING)
    else:
        return _colorize(f"{score:.3f}", Colors.ERROR)


def format_ratio(numerator, denominator):
    """Format ratio with color based on completeness."""
    if denominator == 0:
        return _colorize("N/A", Colors.DIM)

    ratio = numerator / denominator
    ratio_text = f"{numerator}/{denominator}"

    if ratio >= 0.9:
        return _colorize(ratio_text, Colors.SUCCESS)
    elif ratio >= 0.7:
        return _colorize(ratio_text, Colors.WARNING)
    else:
        return _colorize(ratio_text, Colors.ERROR)

def compare_workflow(generated_workflow, reference_workflow):
    """
    Get a comprehensive summary of the evaluation results with enhanced formatting and colors.

    Returns:
        str: A beautifully formatted summary of the evaluation results
    """
    # Format summary with enhanced colors and formatting
    summary = []

    # Main header
    summary.append(_colorize("=" * 80, Colors.HEADER))
    summary.append(_colorize("WORKFLOW COMPARISON ANALYSIS", Colors.HEADER + Colors.BOLD))
    summary.append(_colorize("=" * 80, Colors.HEADER))
    summary.append("")

    # Detailed structural analysis
    structural_analyzer = StructuralMetric()
    structural = structural_analyzer.workflow_structural_analysis(generated_workflow, reference_workflow)
    summary.append(_colorize("🏗️ STRUCTURAL ANALYSIS", Colors.SECTION + Colors.BOLD))
    summary.append(_colorize("-" * 80, Colors.SUBHEADER))

    # Key metrics with minimal icons
    summary.append(
        f"Subtree match ratio: {format_score(structural.get('subtree_match_ratio', 0))}"
    )
    summary.append(f"Weighted subtree match accuracy: {format_score(structural.get('weighted_subtree_match_accuracy', 0))}")
    summary.append(
        f"Matched Subtrees: {format_ratio(structural['match_count'], structural['total_subtrees'])}"
    )
    summary.append(
        f"Missing Subtrees: {format_ratio(structural['miss_count'], structural['total_subtrees'])}"
    )
    summary.append("")

    # Node type breakdown with better formatting
    if "detailed_breakdown" in structural:
        breakdown = structural["detailed_breakdown"]
        summary.append(_colorize("Node Type Analysis", Colors.SUBHEADER + Colors.BOLD))
        summary.append("")

        # Action nodes
        action_matched = breakdown["action_nodes_matched"]
        action_missing = breakdown["action_nodes_missing"]
        action_total = action_matched + action_missing
        summary.append(f"  Action Nodes: {format_ratio(action_matched, action_total)} matched")
        if action_missing > 0:
            summary.append(f"    └─ {_colorize(f'{action_missing} missing', Colors.ERROR)}")

        # Container nodes
        container_matched = breakdown["container_nodes_matched"]
        container_missing = breakdown["container_nodes_missing"]
        container_total = container_matched + container_missing
        summary.append(
            f"  Container Nodes: {format_ratio(container_matched, container_total)} matched"
        )
        if container_missing > 0:
            summary.append(f"    └─ {_colorize(f'{container_missing} missing', Colors.ERROR)}")

        # Depth analysis
        deep_matched = breakdown["deep_structures_matched"]
        shallow_matched = breakdown["shallow_structures_matched"]
        summary.append(
            f"  Deep Structures (≥2): {_colorize(str(deep_matched), Colors.VALUE)} matched"
        )
        summary.append(
            f"  Shallow Structures (<2): {_colorize(str(shallow_matched), Colors.VALUE)} matched"
        )
        summary.append("")

    # Missing structural patterns with detailed breakdown
    if structural["miss_count"] > 0:
        summary.append(_colorize("❌ MISSING STRUCTURAL PATTERNS", Colors.ERROR + Colors.BOLD))
        summary.append(_colorize("-" * 80, Colors.ERROR))

        missing_by_type = {}
        for miss in structural["misses"]:
            node_type = miss["type"]
            if node_type not in missing_by_type:
                missing_by_type[node_type] = []
            missing_by_type[node_type].append(miss)

        for node_type, misses in missing_by_type.items():
            type_name = node_type.replace("_", " ").title()
            summary.append(
                f"Missing {type_name} Nodes: {_colorize(str(len(misses)), Colors.VALUE)}"
            )

            for i, miss in enumerate(misses[:5]):  # Show first 5 examples
                depth_info = f"depth {miss['depth']}"
                children_info = (
                    f"{miss['children_count']} children"
                    if miss["children_count"] > 0
                    else "leaf node"
                )
                prefix = "├─" if i < min(2, len(misses) - 1) else "└─"
                summary.append(
                    f"  {prefix} {_colorize(depth_info, Colors.DIM)}, {_colorize(children_info, Colors.DIM)}"
                )

            if len(misses) > 5:
                summary.append(
                    f"  └─ {_colorize(f'... and {len(misses) - 5} more', Colors.DIM)}"
                )
            summary.append("")

    # Semantic analysis with reduced emphasis but clear presentation
    semantic_analyzer = SemanticMetric()
    semantic = semantic_analyzer.workflow_semantic_analysis(generated_workflow, reference_workflow)
    summary.append(_colorize("🧠 SEMANTIC ANALYSIS", Colors.SECTION + Colors.BOLD))
    summary.append(_colorize("-" * 80, Colors.SUBHEADER))

    # Tool call analysis
    summary.append(f"Average tool call similarity: {format_score(semantic.get('average_tool_call_similarity', 0))}")
    
    # Individual tool call similarities
    if semantic.get('tool_call_similarity'):
        tool_similarities = semantic['tool_call_similarity']
        summary.append(f"Tool-specific similarities:")
        for tool_name, similarity in tool_similarities.items():
            summary.append(f"  └─ {_colorize(tool_name, Colors.VALUE)}: {format_score(similarity)}")
    
    # Missing tools
    if semantic.get('missing_tools'):
        missing_tools = semantic['missing_tools']
        summary.append(f"Missing tools: {_colorize(str(len(missing_tools)), Colors.ERROR)}")
        for tool in missing_tools[:5]:  # Show first 5 missing tools
            summary.append(f"  └─ {_colorize(tool, Colors.ERROR)}")
        if len(missing_tools) > 5:
            summary.append(f"  └─ {_colorize(f'... and {len(missing_tools) - 5} more', Colors.DIM)}")
    
    summary.append("")

    # Variable analysis
    summary.append(_colorize("Variable Analysis", Colors.SUBHEADER + Colors.BOLD))
    summary.append(f"Variable definition similarity: {format_score(semantic.get('average_variable_definition_similarity', 0))}")
    summary.append(f"Variable usage similarity: {format_score(semantic.get('average_variable_usage_similarity', 0))}")
    
    # Individual variable usage similarities
    if semantic.get('variable_usage_similarity'):
        var_similarities = semantic['variable_usage_similarity']
        if var_similarities:
            summary.append(f"Variable-specific usage similarities:")
            for var_name, similarity in var_similarities.items():
                summary.append(f"  └─ {_colorize(var_name, Colors.VALUE)}: {format_score(similarity)}")
    
    # Missing variables
    if semantic.get('missing_variables'):
        missing_vars = semantic['missing_variables']
        summary.append(f"Missing variables: {_colorize(str(len(missing_vars)), Colors.ERROR)}")
        for var in missing_vars[:5]:  # Show first 5 missing variables
            summary.append(f"  └─ {_colorize(var, Colors.ERROR)}")
        if len(missing_vars) > 5:
            summary.append(f"  └─ {_colorize(f'... and {len(missing_vars) - 5} more', Colors.DIM)}")
    
    summary.append("")

    # Final summary section
    summary.append(_colorize("📊 OVERALL ASSESSMENT", Colors.HEADER + Colors.BOLD))
    summary.append(_colorize("-" * 80, Colors.HEADER))
    
    # Calculate overall scores
    structural_score = structural.get('weighted_subtree_match_accuracy', 0)
    semantic_score = (semantic.get('average_tool_call_similarity', 0) + 
                        semantic.get('average_variable_definition_similarity', 0) + 
                        semantic.get('average_variable_usage_similarity', 0)) / 3
    
    summary.append(f"Structural Score: {format_score(structural_score)}")
    summary.append(f"Semantic Score: {format_score(semantic_score)}")
    summary.append(f"Combined Score: {format_score((structural_score + semantic_score) / 2)}")
    
    summary.append("")
    summary.append(_colorize("=" * 80, Colors.HEADER))

    return "\n".join(summary)
