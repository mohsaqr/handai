"""
Tool Registry
Central registry for managing and discovering tools
"""

from typing import Dict, List, Optional, Type
from .base import BaseTool


class ToolRegistry:
    """
    Central registry for Handai tools.

    Usage:
        # Register a tool
        ToolRegistry.register(TransformTool())

        # Get a tool by ID
        tool = ToolRegistry.get_tool("transform")

        # List all tools
        tools = ToolRegistry.list_tools()

        # Get tools by category
        tools = ToolRegistry.get_tools_by_category("Processing")
    """

    _tools: Dict[str, BaseTool] = {}
    _categories: Dict[str, List[str]] = {}

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool instance to register
        """
        if tool.id in cls._tools:
            raise ValueError(f"Tool with ID '{tool.id}' is already registered")

        cls._tools[tool.id] = tool

        # Track by category
        if tool.category not in cls._categories:
            cls._categories[tool.category] = []
        cls._categories[tool.category].append(tool.id)

    @classmethod
    def unregister(cls, tool_id: str) -> None:
        """
        Remove a tool from the registry.

        Args:
            tool_id: ID of tool to remove
        """
        if tool_id in cls._tools:
            tool = cls._tools[tool_id]
            del cls._tools[tool_id]

            # Remove from category
            if tool.category in cls._categories:
                cls._categories[tool.category].remove(tool_id)
                if not cls._categories[tool.category]:
                    del cls._categories[tool.category]

    @classmethod
    def get_tool(cls, tool_id: str) -> Optional[BaseTool]:
        """
        Get a tool by its ID.

        Args:
            tool_id: ID of the tool

        Returns:
            Tool instance or None if not found
        """
        return cls._tools.get(tool_id)

    @classmethod
    def list_tools(cls) -> List[BaseTool]:
        """
        Get list of all registered tools.

        Returns:
            List of tool instances
        """
        return list(cls._tools.values())

    @classmethod
    def list_tool_ids(cls) -> List[str]:
        """
        Get list of all registered tool IDs.

        Returns:
            List of tool IDs
        """
        return list(cls._tools.keys())

    @classmethod
    def get_tools_by_category(cls, category: str) -> List[BaseTool]:
        """
        Get all tools in a category.

        Args:
            category: Category name

        Returns:
            List of tools in the category
        """
        tool_ids = cls._categories.get(category, [])
        return [cls._tools[tid] for tid in tool_ids if tid in cls._tools]

    @classmethod
    def get_categories(cls) -> List[str]:
        """
        Get list of all categories.

        Returns:
            List of category names
        """
        return list(cls._categories.keys())

    @classmethod
    def get_tool_info(cls) -> List[Dict]:
        """
        Get information about all registered tools.

        Returns:
            List of tool info dictionaries
        """
        return [tool.get_info() for tool in cls._tools.values()]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools"""
        cls._tools.clear()
        cls._categories.clear()

    @classmethod
    def tool_exists(cls, tool_id: str) -> bool:
        """Check if a tool is registered"""
        return tool_id in cls._tools


def register_default_tools():
    """Register the default Handai tools"""
    from .transform import TransformTool
    from .generate import GenerateTool
    from .process_documents import ProcessDocumentsTool
    from .qualitative import QualitativeTool
    from .consensus import ConsensusTool
    from .codebook_generator import CodebookGeneratorTool
    from .automator import AutomatorTool
    from .manual_coder import ManualCoderTool

    # Only register if not already registered
    if not ToolRegistry.tool_exists("transform"):
        ToolRegistry.register(TransformTool())

    if not ToolRegistry.tool_exists("generate"):
        ToolRegistry.register(GenerateTool())

    if not ToolRegistry.tool_exists("process_documents"):
        ToolRegistry.register(ProcessDocumentsTool())

    if not ToolRegistry.tool_exists("qualitative"):
        ToolRegistry.register(QualitativeTool())

    if not ToolRegistry.tool_exists("consensus"):
        ToolRegistry.register(ConsensusTool())

    if not ToolRegistry.tool_exists("codebook-generator"):
        ToolRegistry.register(CodebookGeneratorTool())

    if not ToolRegistry.tool_exists("automator"):
        ToolRegistry.register(AutomatorTool())

    if not ToolRegistry.tool_exists("manual_coder"):
        ToolRegistry.register(ManualCoderTool())
