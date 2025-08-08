#!/usr/bin/env python3
"""
Screenshot-friendly workflow demonstration.

This script shows the most compact version of a workflow that includes
all schema node types in a flat, easy-to-screenshot format.
"""

import json
from pathlib import Path
from visualizer.workflow_visualizer import WorkflowVisualizer
from visualizer.tools_visualizer import ToolsVisualizer

def main():
    """Demonstrate the screenshot-friendly workflow."""
    
    current_dir = Path(__file__).parent
    tools_file = current_dir / "data_schema" / "demo_tools_list.json"
    try:
        with open(tools_file, 'r', encoding='utf-8') as f:
            tools_data = json.load(f)
    except Exception as e:
        print(f"Error loading tools: {e}")
        return
    # Create tools visualizer
    tools_visualizer = ToolsVisualizer(
        indent_size=4,
        use_colors=True,
        use_icons=True  
    )
    # Visualize tools
    tools_visualization = tools_visualizer.visualize_tools(tools_data)
    print(tools_visualization)

    workflow_file = current_dir / "data_schema" / "demo_workflow_dict.json"
    try:
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
    except Exception as e:
        print(f"Error loading workflow: {e}")
        return
    # Create visualizer
    visualizer = WorkflowVisualizer(
        indent_size=4,
        use_colors=True,
        use_icons=True
    )
    # Visualize
    visualization = visualizer.visualize_workflow(workflow_data)
    print(visualization)

if __name__ == "__main__":
    main()
