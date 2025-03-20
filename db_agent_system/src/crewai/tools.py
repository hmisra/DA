"""
Tool adapters for converting LangChain tools to CrewAI-compatible tools.
This module provides adapters that make LangChain tools compatible with CrewAI agents.
"""
from typing import List, Any, Dict, Optional, Type
import json
from pydantic import Field, create_model
from langchain_core.tools import BaseTool as LangChainBaseTool
from crewai.tools import BaseTool as CrewAIBaseTool
from src.utils.logging import AgentLogger


class LangChainToolAdapter(CrewAIBaseTool):
    """Adapter that converts a LangChain tool to a CrewAI-compatible tool."""

    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of the tool")
    langchain_tool: Any = Field(description="The LangChain tool being adapted")
    
    model_config = {"extra": "allow"}
    
    def __init__(self, lc_tool: LangChainBaseTool):
        """Initialize the adapter with a LangChain tool.

        Args:
            lc_tool: The LangChain tool to adapt
        """
        # Extract properties from the LangChain tool
        tool_name = getattr(lc_tool, "name", "Unknown Tool")
        tool_description = getattr(lc_tool, "description", "No description provided")
        
        # Initialize with parameters
        super().__init__(
            name=tool_name,
            description=tool_description,
            langchain_tool=lc_tool
        )

    def _run(self, tool_input: str) -> Any:
        """Execute the LangChain tool with the parsed input.

        Args:
            tool_input: A string input that will be parsed as JSON if needed

        Returns:
            Any: The result of the tool execution
        """
        logger = AgentLogger(agent_name=f"ToolAdapter-{self.name}", agent_type="CrewAI Tool")
        
        logger.info(f"Executing tool: {self.name}")
        logger.info(f"Tool input: {str(tool_input)[:200]}...")
        
        try:
            # Parse the input - try as JSON first, fallback to string
            if isinstance(tool_input, str):
                try:
                    # The crewAI agent typically sends a JSON string
                    kwargs = json.loads(tool_input)
                    logger.info(f"Parsed JSON input: {str(kwargs)}")
                except json.JSONDecodeError:
                    # If not JSON, use as a direct input
                    logger.info("Failed to parse as JSON, using as direct input")
                    kwargs = {"input": tool_input}
            elif isinstance(tool_input, dict):
                kwargs = tool_input
                logger.info(f"Using dictionary input: {str(kwargs)}")
            else:
                error_msg = f"Unsupported input type: {type(tool_input)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Check if we have the required args_schema and adapt accordingly
            if hasattr(self.langchain_tool, "args_schema"):
                # Get expected args from schema
                schema_fields = self.langchain_tool.args_schema.__annotations__
                logger.info(f"Tool schema fields: {schema_fields}")
                
                # For positional args methods like GetTableSchemaTool._run(table_name)
                # Check the actual method signature
                if hasattr(self.langchain_tool, "_run"):
                    method = self.langchain_tool._run
                    arg_key = next(iter(schema_fields), None)
                    
                    # If we have a single arg in schema and it's in kwargs
                    if len(schema_fields) == 1 and arg_key in kwargs:
                        # Special case for tools that expect positional args
                        logger.info(f"Executing with positional arg: {arg_key}={kwargs[arg_key]}")
                        result = self.langchain_tool._run(kwargs[arg_key])
                        logger.info(f"Tool execution result: {str(result)[:200]}...")
                        return result
            
            # Standard execution with kwargs
            if hasattr(self.langchain_tool, "_run"):
                logger.info(f"Executing with kwargs: {str(kwargs)}")
                result = self.langchain_tool._run(**kwargs)
                logger.info(f"Tool execution result: {str(result)[:200]}...")
                return result
            elif hasattr(self.langchain_tool, "run"):
                logger.info(f"Executing run method with kwargs: {str(kwargs)}")
                result = self.langchain_tool.run(**kwargs)
                logger.info(f"Tool execution result: {str(result)[:200]}...")
                return result
            else:
                error_msg = "LangChain tool has no _run or run method"
                logger.error(error_msg)
                raise AttributeError(error_msg)
        except Exception as e:
            error_msg = f"Error executing tool {self.name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg


def adapt_langchain_tools(tools: List[LangChainBaseTool]) -> List[CrewAIBaseTool]:
    """Convert a list of LangChain tools to CrewAI-compatible tools.

    Args:
        tools: List of LangChain tools

    Returns:
        List[CrewAIBaseTool]: List of CrewAI-compatible tools
    """
    return [LangChainToolAdapter(lc_tool=tool) for tool in tools] 