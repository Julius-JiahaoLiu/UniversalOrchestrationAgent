"""
Schema utilities for the Elastic Gumby Universal Orchestration Agent.

This module provides centralized access to JSON schemas used throughout the application,
including workflow orchestration schema and tools schema.
"""

import json
from pathlib import Path
from typing import Any, Dict

def get_workflow_schema() -> Dict[str, Any]:
    """
    Get the workflow orchestration schema.

    Returns:
        Dict[str, Any]: The workflow schema dictionary.
    """
    _schema_dir = Path(__file__).parent
    schema_path = _schema_dir / "workflow_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        _schema = json.load(f)
    return _schema


def get_tools_schema() -> Dict[str, Any]:
    """
    Get the tools schema.

    Returns:
        Dict[str, Any]: The tools schema dictionary.
    """
    _schema_dir = Path(__file__).parent
    schema_path = _schema_dir / "tools_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        _schema = json.load(f)
    return _schema
