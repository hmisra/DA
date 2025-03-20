"""
Register AgentIQ plugins.
This module makes it easy to register our plugins with AgentIQ.
"""

from src.agentiq.plugins.ollama.ollama_llm import ollama_llm


def register_llm_providers():
    """Register LLM providers with AgentIQ."""
    return {
        "ollama": ollama_llm
    } 