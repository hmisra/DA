"""
Adapter for connecting Ollama with AgentIQ.
This module implements the bridge between our existing Ollama implementation and the AgentIQ framework.
"""
from typing import Dict, List, Any, Optional, AsyncGenerator
from abc import ABC, abstractmethod
import asyncio

# Import langchain-ollama directly
from langchain_ollama import OllamaLLM


# Define the AgentIQ LLM client classes
class TokenPart:
    """Represents a token in a streaming response."""
    
    def __init__(self, content: str):
        """Initialize a TokenPart.
        
        Args:
            content: The token content
        """
        self.content = content


class LLMClientResponse:
    """Represents a response from an LLM."""
    
    def __init__(self, content: str):
        """Initialize an LLMClientResponse.
        
        Args:
            content: The response content
        """
        self.content = content


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        stop: Optional[List[str]] = None
    ) -> LLMClientResponse:
        """Generate a response for the given prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            stop: Optional stop sequences
            
        Returns:
            LLMClientResponse: The generated response
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        stop: Optional[List[str]] = None
    ) -> AsyncGenerator[TokenPart, None]:
        """Stream a response for the given prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            stop: Optional stop sequences
            
        Yields:
            TokenPart: Stream of token parts
        """
        pass


class OllamaLLMClient(LLMClient):
    """Implementation of LLMClient for Ollama using langchain-ollama."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the OllamaLLMClient.
        
        Args:
            config: The configuration dictionary
        """
        # Extract parameters from config
        model_name = config.get("model_name", "llama3")
        base_url = config.get("base_url", "http://localhost:11434")
        temperature = config.get("temperature", 0.0)
        top_p = config.get("top_p", 1.0)
        num_ctx = config.get("num_ctx", 4096)
        
        # Create langchain-ollama instance directly
        self.ollama_llm = OllamaLLM(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            num_ctx=num_ctx
        )
        
        # Save the config for logging
        self.config = config
        print(f"Initialized OllamaLLMClient with model {model_name} at {base_url}")

    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        stop: Optional[List[str]] = None
    ) -> LLMClientResponse:
        """Generate a response for the given prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            stop: Optional stop sequences
            
        Returns:
            LLMClientResponse: The generated response
        """
        print(f"Generating response for prompt: {prompt[:100]}...")
        
        # Prepare kwargs
        kwargs = {}
        if stop:
            kwargs["stop"] = stop
        
        # Handle system prompt if provided
        if system_prompt:
            # For Ollama, we need to prepend the system prompt to the user prompt
            # as langchain-ollama doesn't have a separate system prompt param in the basic version
            prompt = f"{system_prompt}\n\n{prompt}"
        
        # Call langchain-ollama implementation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: self.ollama_llm.invoke(prompt, **kwargs)
        )
        
        print(f"Response generated, length: {len(response)}")
        
        # Return response in AgentIQ format
        return LLMClientResponse(content=response)

    async def generate_stream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        stop: Optional[List[str]] = None
    ) -> AsyncGenerator[TokenPart, None]:
        """Stream a response for the given prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            stop: Optional stop sequences
            
        Yields:
            TokenPart: Stream of token parts
        """
        print(f"Streaming response for prompt: {prompt[:100]}...")
        
        # Prepare kwargs
        kwargs = {}
        if stop:
            kwargs["stop"] = stop
            
        # Handle system prompt if provided
        if system_prompt:
            # For Ollama, we need to prepend the system prompt to the user prompt
            prompt = f"{system_prompt}\n\n{prompt}"
        
        # Get the stream generator - need to run in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        stream_gen = await loop.run_in_executor(
            None,
            lambda: list(self.ollama_llm.stream(prompt, **kwargs))
        )
        
        # Convert to async generator
        for chunk in stream_gen:
            yield TokenPart(content=chunk) 