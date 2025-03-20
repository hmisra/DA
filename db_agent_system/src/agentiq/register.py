"""
Register AgentIQ components.
This module registers the database tools and workflow with AgentIQ.
"""

from src.agentiq.workflow import DBQueryWorkflow, create_db_query_workflow
from src.langchain.database import get_db_tools
from src.agentiq.plugins.ollama.ollama_llm import ollama_llm


def register_workflow():
    """Register the workflow with AgentIQ."""
    return {
        "db_query_workflow": create_db_query_workflow
    }


def register_tools():
    """Register the database tools with AgentIQ."""
    tools_map = {}
    for tool in get_db_tools():
        tools_map[tool.name] = tool
    return tools_map


def register_llm_providers():
    """Register LLM providers with AgentIQ."""
    return {
        "ollama": ollama_llm
    } 