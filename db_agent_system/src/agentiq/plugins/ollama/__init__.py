"""
Ollama LLM provider plugin for NVIDIA AgentIQ.
"""

from .ollama_llm import OllamaModelConfig, ollama_llm
from .adapter import OllamaLLMClient

__all__ = ["OllamaModelConfig", "ollama_llm", "OllamaLLMClient"] 