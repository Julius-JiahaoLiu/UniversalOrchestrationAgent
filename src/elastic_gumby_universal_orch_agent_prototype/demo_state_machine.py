
from pathlib import Path
import json
import subprocess
from elastic_gumby_universal_orch_agent_prototype.transform.state_machine_transformer import StateMachineTransformer
from elastic_gumby_universal_orch_agent_prototype.visualizer.workflow_loader import WorkflowLoader

def main():
    """Main function to demonstrate workflow transformation and execution."""
    
    current_dir = Path(__file__).parent
    # Load generated workflow
    generated_workflow_file = current_dir / "data_test" / "test_QT_workflow.json"
    with open(generated_workflow_file, 'r', encoding='utf-8') as f:
        generated_workflow = json.load(f)
    available_tools_file = current_dir / "data_test" / "reference_QT_tools.json"
    with open(available_tools_file, 'r', encoding='utf-8') as f:
        available_tools = json.load(f)

    tools_definition = {tool['name']: tool['parameters'] for tool in available_tools["available_tools"]}
    workflow_loader = WorkflowLoader(use_colors=True, tools_definition=tools_definition)
    result = workflow_loader.load_workflow_from_json_string(json.dumps(generated_workflow))
    if result["success"]:
        transformer = StateMachineTransformer(available_tools["available_tools"])
        transformer.save_state_machine(generated_workflow, current_dir)



if __name__ == "__main__":
    main()