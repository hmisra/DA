"""
Ollama LLM provider for NVIDIA AgentIQ.
This module implements the Ollama LLM provider for AgentIQ.
"""
from typing import Dict, Any, AsyncGenerator
from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field

from aiq.builder.llm import LLMProviderInfo
from aiq.cli.register_workflow import register_llm_provider
from aiq.data_models.llm import LLMBaseConfig
from .adapter import OllamaLLMClient


class OllamaModelConfig(LLMBaseConfig, name="ollama"):
    """An Ollama LLM provider to be used with an LLM client."""

    model_config = ConfigDict(protected_namespaces=())

    base_url: str = Field(default="http://localhost:11434", description="Base URL for the Ollama API.")
    model_name: str = Field(validation_alias=AliasChoices("model_name", "model"),
                            serialization_alias="model",
                            description="The Ollama model name to use.")
    temperature: float = Field(default=0.0, description="Sampling temperature in [0, 1].")
    top_p: float = Field(default=1.0, description="Top-p for distribution sampling.")
    num_ctx: int = Field(default=4096, description="The context window size in tokens.")
    num_predict: int | None = Field(default=None, description="Maximum number of tokens to predict.")
    system_prompt: str | None = Field(default=None, description="Optional system prompt to control model behavior.")
    options: dict | None = Field(default=None, description="Additional model parameters as dictionary.")


@register_llm_provider(config_type=OllamaModelConfig)
async def ollama_llm(config: OllamaModelConfig, **kwargs) -> AsyncGenerator[LLMProviderInfo, None]:
    """Register the Ollama LLM provider.
    
    Args:
        config: The Ollama model configuration
        **kwargs: Additional arguments (provided by AgentIQ plugin system)
        
    Returns:
        AsyncGenerator[LLMProviderInfo, None]: The LLM provider info
    """
    # Create our LLM client
    llm_client = OllamaLLMClient(config.model_dump())
    
    # Return provider info with our client
    yield LLMProviderInfo(
        config=config, 
        description="An Ollama model for use with an LLM client.",
        llm_client=llm_client
    ) 