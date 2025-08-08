"""
Base classes and utilities for visualizers.

This module contains common color codes, icons, and base functionality
shared across different visualizer implementations.
"""

import re
from typing import Any

class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Node type colors
    TOOL_CALL = "\033[94m"  # Blue
    USER_INPUT = "\033[93m"  # Yellow
    BRANCH = "\033[95m"  # Magenta
    LOOP = "\033[96m"  # Cyan
    WAIT_EVENT = "\033[91m"  # Red
    SEQUENCE = "\033[92m"  # Green
    PARALLEL = "\033[97m"  # White

    # Special colors
    VARIABLE = "\033[33m"  # Orange/Yellow
    DESCRIPTION = "\033[90m"  # Gray
    WORKFLOW_TITLE = "\033[1;36m"  # Bold Cyan
    OUTPUT_VAR = "\033[32m"  # Green

    # Colorama-style colors for tools visualization
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    LIGHTBLACK_EX = "\033[90m"


class Icons:
    """Unicode icons for different workflow and tool components."""

    # Workflow icons
    WORKFLOW = "🔄"
    TOOL_CALL = "🔧"
    USER_INPUT = "👤"
    BRANCH = "🔀"
    LOOP = "🔁"
    WAIT_EVENT = "⏳"
    SEQUENCE = "➡️"
    PARALLEL = "⏩"
    INPUT = "📥"
    CONDITION = "❓"
    TRUE = "✅"
    FALSE = "❌"
    TIMEOUT = "⏰"
    DESCRIPTION = "📝"

    # Tools icons
    TOOLS_HEADER = "🔧"
    TOOL_ITEM = "🛠️"
    PARAMETERS = "📊"
    RETURNS = "📤"
    METADATA = "ℹ️"
    REQUIRED = "●"
    OPTIONAL = "○"
    SOURCE_JSON = "📄"
    SOURCE_RAW = "✏️"
    NO_TOOLS = "📭"


class BaseVisualizer:
    """Base class for all visualizers with common functionality."""

    def __init__(self, indent_size: int = 4, use_colors: bool = True, use_icons: bool = True):
        """Initialize the base visualizer.

        Args:
            indent_size: Number of spaces for each indentation level
            use_colors: Whether to use ANSI colors in output
            use_icons: Whether to use Unicode icons in output
        """
        self.indent_size = indent_size
        self.indent_char = " "
        self.use_colors = use_colors
        self.use_icons = use_icons
        self.branch_chars = {"pipe": "│", "tee": "├──", "last": "└──", "space": " " * 3}

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text

    def _iconize(self, icon: str) -> str:
        """Add icon if icons are enabled."""
        if self.use_icons:
            return f"{icon} "
        return ""

    def _highlight_variables(self, text: str) -> str:
        """Highlight variable references in the format ${variableName}."""
        if not self.use_colors:
            return text

        pattern = r"(\$[a-zA-Z_][\w\.]*)"  # Match $variable, $variable.property in group 1

        def replace_var(match: Any) -> str:
            var_name = match.group(1)
            return self._colorize(var_name, Colors.VARIABLE)

        return re.sub(pattern, replace_var, text)

    def _strip_ansi_codes(self, text: str) -> str:
        """Remove ANSI color codes from text for clean file output."""
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)