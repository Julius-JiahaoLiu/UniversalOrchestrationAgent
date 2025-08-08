
import json
from pathlib import Path
from metrics.utils import compare_workflow

def main():
    """Main function to demonstrate workflow comparison."""
    
    current_dir = Path(__file__).parent
    # Load generated workflow
    generated_workflow_file = current_dir / "data_test" / "generated_short_workflow.json"
    try:
        with open(generated_workflow_file, 'r', encoding='utf-8') as f:
            generated_workflow = json.load(f)
    except Exception as e:
        print(f"Error loading generated workflow: {e}")
        return 
    # Load reference workflow
    reference_workflow_file = current_dir / "data_test" / "reference_short_workflow.json"
    try:
        with open(reference_workflow_file, 'r', encoding='utf-8') as f:
            reference_workflow = json.load(f)
    except Exception as e:
        print(f"Error loading reference workflow: {e}")
        return
    
    # Compare workflows
    summary = compare_workflow(generated_workflow, reference_workflow)
    print(summary)

if __name__ == "__main__":
    main()