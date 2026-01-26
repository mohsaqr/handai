"""
Base Tool Abstract Class
Foundation for extensible tool system
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class ToolConfig:
    """Configuration returned by a tool's render_config method"""
    is_valid: bool
    error_message: Optional[str] = None
    config_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.config_data is None:
            self.config_data = {}


@dataclass
class ToolResult:
    """Result returned by a tool's execute method"""
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    stats: Dict[str, Any] = None

    def __post_init__(self):
        if self.stats is None:
            self.stats = {}


class BaseTool(ABC):
    """
    Abstract base class for Handai tools.

    To create a new tool:
    1. Subclass BaseTool
    2. Set class attributes (id, name, description, icon)
    3. Implement render_config() to render UI and return config
    4. Implement execute() to perform the tool's action
    5. Register with ToolRegistry.register(YourTool())
    """

    # Tool metadata - override in subclass
    id: str = "base"
    name: str = "Base Tool"
    description: str = "Base tool description"
    icon: str = ""
    category: str = "General"

    @abstractmethod
    def render_config(self) -> ToolConfig:
        """
        Render tool-specific configuration UI using Streamlit.

        This method should:
        1. Render any UI elements for configuration (file uploads, inputs, etc.)
        2. Validate the configuration
        3. Return a ToolConfig with:
           - is_valid: True if configuration is complete and valid
           - error_message: Error message if not valid
           - config_data: Dictionary with all config values needed for execute()

        Returns:
            ToolConfig with configuration data
        """
        pass

    @abstractmethod
    async def execute(
        self,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> ToolResult:
        """
        Execute the tool with the given configuration.

        This method should:
        1. Perform the tool's main action (processing, generation, etc.)
        2. Call progress_callback periodically if provided
        3. Return a ToolResult with:
           - success: True if execution completed successfully
           - data: The result data (DataFrame, dict, etc.)
           - error_message: Error message if not successful
           - stats: Dictionary with statistics (counts, timing, etc.)

        Args:
            config: Configuration dictionary from render_config()
            progress_callback: Optional callback for progress updates
                              signature: (completed, total, success, errors, retries, log_entry, is_error)

        Returns:
            ToolResult with execution results
        """
        pass

    def render_results(self, result: ToolResult):
        """
        Render the results of tool execution.

        Override this method to customize result display.
        Default implementation shows basic stats and data preview.

        Args:
            result: ToolResult from execute()
        """
        import streamlit as st

        if not result.success:
            st.error(f"Execution failed: {result.error_message}")
            return

        # Show stats if available
        if result.stats:
            cols = st.columns(len(result.stats))
            for col, (key, value) in zip(cols, result.stats.items()):
                with col:
                    st.metric(key.replace("_", " ").title(), value)

        # Show data preview if it's a DataFrame
        if result.data is not None:
            import pandas as pd
            if isinstance(result.data, pd.DataFrame):
                st.dataframe(result.data, use_container_width=True)

    def get_info(self) -> Dict[str, str]:
        """Get tool information as dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "category": self.category
        }

    def __repr__(self):
        return f"<{self.__class__.__name__} id='{self.id}' name='{self.name}'>"
