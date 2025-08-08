"""
Phases Module

This module contains the phase handlers for the
Universal Transformation Orchestration Agent (UTOA).

Each phase represents a distinct stage in the workflow processing:
- Phase 1: Tools Onboarding - Collect and process tool definitions
- Phase 2: Planning & Reflection - Generate and refine workflow plans
- Phase 3: Execution & Monitor - Execute workflows with fallback strategies
"""

from .phase1_tools_onboarding import Phase1ToolsOnboarding
from .phase2_planning_reflecting import Phase2PlanningReflecting
from .phase3_transform_execution import Phase3TransformExecution

__all__ = ["Phase1ToolsOnboarding", "Phase2PlanningReflecting", "Phase3TransformExecution"]

__version__ = "1.0.0"
