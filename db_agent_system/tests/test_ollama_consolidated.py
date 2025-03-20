"""
Test module for Ollama integration.
This consolidated test covers all Ollama-related functionality.
"""
import pytest
import asyncio

from conftest import BaseTestClass, ollama_config
from src.ollama.llm import OllamaManager, get_ollama_llm
from src.agentiq.plugins.ollama.adapter import OllamaLLMClient, LLMClientResponse, TokenPart
from src.agentiq.plugins.ollama.ollama_llm import OllamaModelConfig


class TestOllamaCore(BaseTestClass):
    """Test core Ollama functionality."""
    
    def test_ollama_connection(self):
        """Test connection to Ollama."""
        # Initialize OllamaManager
        ollama_mgr = OllamaManager()
        
        # Get all available models
        models = ollama_mgr.get_all_available_models()
        assert models, "No Ollama models available"
        print(f"\nAvailable Ollama models: {', '.join(models)}")
        
        # Select best model for task
        model_name = ollama_mgr.select_best_model("Generate SQL queries")
        assert model_name, "No model selected"
        print(f"\nSelected model for SQL queries: {model_name}")
        
        # Get LLM
        llm = get_ollama_llm(model_name)
        assert llm is not None, "Failed to get Ollama LLM"
        
        # Test LLM
        response = llm("What is the capital of France?")
        assert response, "No response from LLM"
        print(f"\nLLM Response: \n{response[:100]}...")


class TestOllamaPlugin(BaseTestClass):
    """Test the AgentIQ Ollama plugin."""
    
    @pytest.mark.asyncio
    async def test_ollama_client_creation(self, ollama_config):
        """Test that the OllamaLLMClient can be created from a config."""
        # Create an Ollama client
        client = OllamaLLMClient(ollama_config)
        
        # Verify that the client was initialized with the correct parameters
        assert client.ollama_llm.model == "llama3"
        assert client.ollama_llm.base_url == "http://localhost:11434"
        assert client.ollama_llm.temperature == 0.1
        assert client.ollama_llm.top_p == 0.9
        assert client.ollama_llm.num_ctx == 2048
        
        print("OllamaLLMClient created successfully with the correct parameters")
    
    @pytest.mark.asyncio
    async def test_ollama_simple_query(self, ollama_config):
        """Test running a simple query with the Ollama LLM client."""
        # Create a client
        client = OllamaLLMClient(ollama_config)
        
        # Test a simple query using generate_text
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        response = client.generate_text(messages, stream=False)
        
        # Verify response
        assert isinstance(response, LLMClientResponse)
        assert len(response.text) > 0
        print(f"Simple query response: {response.text[:100]}...")
    
    @pytest.mark.asyncio
    async def test_ollama_text_generation_with_system_prompt(self, ollama_config):
        """Test that the Ollama client can generate text with a system prompt."""
        # Create a client
        client = OllamaLLMClient(ollama_config)
        
        # Test text generation with system prompt
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that specializes in SQL."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        response = client.generate_text(messages, stream=False)
        
        # Verify that the response contains meaningful content
        assert isinstance(response, LLMClientResponse)
        assert len(response.text) > 0
        print(f"Response with system prompt: {response.text[:100]}...")
    
    @pytest.mark.asyncio
    async def test_ollama_generate_method(self, ollama_config):
        """Test the generate and generate_stream methods."""
        # Create a client
        client = OllamaLLMClient(ollama_config)
        
        # Test async generation
        prompt = "Count from 1 to 5."
        
        # Using generate method
        response = await client.generate(prompt)
        assert isinstance(response, LLMClientResponse)
        assert len(response.content) > 0
        print(f"Generate response: {response.content[:100]}...")
        
        # Using generate with system prompt
        response_with_system = await client.generate(
            prompt, 
            system_prompt="You are a helpful math teacher."
        )
        assert isinstance(response_with_system, LLMClientResponse)
        assert len(response_with_system.content) > 0
        print(f"Generate with system prompt: {response_with_system.content[:100]}...")
    
    @pytest.mark.asyncio
    async def test_ollama_streaming(self, ollama_config):
        """Test that the Ollama client can generate streaming responses."""
        # Create a client
        client = OllamaLLMClient(ollama_config)
        
        # Test streaming generation
        prompt = "Count from 1 to 5."
        
        # Collect streaming chunks using generate_stream
        chunks = []
        async for chunk in client.generate_stream(prompt):
            chunks.append(chunk.content)
            print(".", end="", flush=True)  # Print a dot for each chunk to show progress
        
        # Verify that we got some chunks
        assert len(chunks) > 0
        
        # Join the chunks and verify content
        full_response = "".join(chunks)
        assert len(full_response) > 0
        print(f"\nStreaming response: {full_response[:100]}...")
        
        # The response should contain numbers
        assert any(str(i) in full_response for i in range(1, 6))
        
        # Test streaming using generate_text_stream
        messages = [{"role": "user", "content": "Count from 6 to 10."}]
        text_chunks = []
        async for chunk in client.generate_text_stream(messages):
            text_chunks.append(chunk.text)
            print(".", end="", flush=True)
        
        # Verify text stream
        assert len(text_chunks) > 0
        text_response = "".join(text_chunks)
        assert len(text_response) > 0
        print(f"\nText stream response: {text_response[:100]}...")
        
        # The response should contain numbers
        assert any(str(i) in text_response for i in range(6, 11)) 