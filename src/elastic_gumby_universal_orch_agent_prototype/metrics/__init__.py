"""
Metrics package for workflow comparison and analysis.

This package provides structural and semantic metrics for comparing workflows
defined in the workflow_schema.json format.

Main Interface:
    WorkflowComparator: The primary interface class with get_summary() method
    
Individual Components:
    SemanticMetric: Semantic analysis of workflows
    StructuralMetric: Structural analysis of workflows
"""



# Import individual metric classes for advanced usage
from .semantic_metric import SemanticMetric
from .structural_metric import StructuralMetric
from .utils import compare_workflow

# Define what gets imported with "from metrics import *"
__all__ = [
    'SemanticMetric',
    'StructuralMetric',
    'compare_workflow',  # Main interface
]
