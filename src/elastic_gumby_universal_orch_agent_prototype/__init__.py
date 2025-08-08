"""
Elastic Gumby Universal Orchestration Agent Prototype

A comprehensive system for workflow orchestration, tool management, and execution
with advanced planning, validation, reflection, and backup capabilities.

Main Components:
- data_schema: Defines schemas for tools and workflows.
- phases: Contains the main phases of the orchestration process.
- planner: Implements the iterative planning and reflection logic.
- transform: Handles the transformation of workflows into executable states.
- visualizer: Provides validation and visualization for workflows and tools against schemas.
- metrics: Implements semantic and structural metrics for workflow comparison.

Usage:
    # import through this package (when installed)
    from elastic_gumby_universal_orch_agent_prototype.phases import Phase1ToolsOnboarding
    from elastic_gumby_universal_orch_agent_prototype.planner import IterativePlanner
"""

# Version information
__version__ = "1.0.0"
__author__ = "Elastic Gumby Team"
__description__ = "Universal Orchestration Agent Prototype"

# Import all components directly - dependencies are properly configured and relative imports fixed
from .agent_main import AgentMainInterface
from .data_schema import get_tools_schema, get_workflow_schema
from .phases import Phase1ToolsOnboarding, Phase2PlanningReflecting, Phase3TransformExecution
from .planner import BedrockClientManager, IterativePlanner, WorkflowProcessor, generate_plan, reflect_plan
from .transform import StateMachineTransformer
from .visualizer import ToolsLoader, ToolsVisualizer, WorkflowLoader, WorkflowVisualizer
from .metrics import SemanticMetric, StructuralMetric, compare_workflow

# Public API - all components are now available
__all__ = [
    # Package metadata
    "__version__",
    "__author__",
    "__description__",
    # Main interface
    "AgentMainInterface",
    # Phases
    "Phase1ToolsOnboarding",
    "Phase2PlanningReflecting",
    "Phase3TransformExecution",
    # Planner components
    "IterativePlanner",
    "generate_plan",
    "reflect_plan",
    "BedrockClientManager",
    "WorkflowProcessor",
    # Transform components
    "StateMachineTransformer",
    # Visualizer components
    "ToolsLoader",
    "WorkflowLoader",
    "ToolsVisualizer",
    "WorkflowVisualizer",
    # Data Schema
    "get_tools_schema",
    "get_workflow_schema",
    "reload_schemas",
    # Metrics
    "SemanticMetric",
    "StructuralMetric",
    "compare_workflow",
]

# Package-level configuration
import logging

# Set up package-level logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Prevent "No handlers" warnings


def get_package_info():
    """Get information about the package and available components."""
    info = {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "public_api": __all__,
    }
    return info
