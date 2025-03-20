"""
Ollama LLM integration module.
Provides a factory for creating LLM instances using Ollama.
"""
import time
import os
from typing import Dict, List, Any, Optional, Union, Callable

from langchain_ollama import OllamaLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Extra, Field, BaseModel, ConfigDict

from src.utils.config import config
from src.utils.logging import AgentLogger


# Create a CrewAI-compatible OllamaLLM
class CrewAICompatibleOllamaLLM(OllamaLLM):
    """OllamaLLM that is compatible with CrewAI by adding the missing methods."""
    
    def supports_stop_words(self) -> bool:
        """Method required by CrewAI for stop token handling.
        
        Returns:
            bool: Whether this LLM supports stop words
        """
        return True
    
    def get_num_tokens(self, text: str) -> int:
        """Method for counting tokens in text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: Approximate token count
        """
        # Rough approximation (4 chars ~ 1 token)
        return len(text) // 4
    
    def build_model_input(
        self, 
        message_str: str, 
        stop: Optional[List[str]] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """Build inputs for the model required by CrewAI.
        
        Args:
            message_str: The message to send
            stop: Optional stop sequences
            
        Returns:
            Dict: The model inputs
        """
        return {
            "prompt": message_str,
            "stop": stop,
            **kwargs
        }
    
    def get_prompt_tokens_size(self, prompt: str) -> int:
        """Get the token size of the prompt.
        
        Args:
            prompt: The prompt to check
            
        Returns:
            int: Approximate token count
        """
        return self.get_num_tokens(prompt)


class OllamaManager:
    """Manager for Ollama LLMs."""

    _instance = None
    _models: Dict[str, "EnhancedOllamaLLM"] = {}
    _logger = None

    def __new__(cls):
        """Create a singleton instance.

        Returns:
            OllamaManager: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(OllamaManager, cls).__new__(cls)
            cls._logger = AgentLogger(
                agent_name="OllamaManager", agent_type="LLM Manager"
            )
            cls._models = {}
        return cls._instance

    def __init__(self):
        """Initialize the OllamaManager."""
        self.logger = AgentLogger(agent_name="OllamaManager", agent_type="LLM")
        
        # Get default model from environment or configuration
        self.default_model = config.OLLAMA_DEFAULT_MODEL if hasattr(config, "OLLAMA_DEFAULT_MODEL") else "llama3.3:latest"
        self.small_model = config.OLLAMA_SMALL_MODEL if hasattr(config, "OLLAMA_SMALL_MODEL") else "llama3.3:latest"
        self.large_model = config.OLLAMA_LARGE_MODEL if hasattr(config, "OLLAMA_LARGE_MODEL") else "llama3.3:latest"

    def get_model(self, model_name: str = None) -> "EnhancedOllamaLLM":
        """Get an LLM instance for the given model name.

        Args:
            model_name: Name of the model to use (default to config)

        Returns:
            EnhancedOllamaLLM: The LLM instance
        """
        if model_name is None:
            model_name = self.default_model

        if model_name not in self._models:
            self._logger.info(f"Creating new Ollama LLM instance for model {model_name}")
            self._models[model_name] = EnhancedOllamaLLM(
                model=model_name,
                base_url=config.OLLAMA_URL,
            )

        return self._models[model_name]

    def get_all_available_models(self) -> List[str]:
        """Get all available models from Ollama.

        Returns:
            List[str]: List of available model names
        """
        import requests
        import json

        self._logger.info("Fetching available models from Ollama")
        try:
            response = requests.get(f"{config.OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                models_data = json.loads(response.text)
                model_names = [model["name"] for model in models_data.get("models", [])]
                self._logger.info(f"Found {len(model_names)} available models")
                return model_names
            else:
                self._logger.error(f"Failed to get models: {response.status_code}")
                return []
        except Exception as e:
            self._logger.error(f"Error fetching models: {str(e)}", exc_info=True)
            return []

    def select_best_model(self, task_description: str, options: List[str] = None) -> str:
        """Select the best model for a given task.

        This implementation uses task-specific models based on the task description.

        Args:
            task_description: Description of the task
            options: Optional list of model names to choose from

        Returns:
            str: Name of the selected model
        """
        self._logger.info(f"Selecting best model for task: {task_description}")
        
        # Force using a smaller model for testing (to make tests faster)
        if os.environ.get("ENV") == "test":
            self._logger.info("Using deepseek-r1:14b model for testing with better tool use capabilities")
            # Use 'deepseek-r1:14b' which has good tool use capabilities
            return "deepseek-r1:14b"

        # Get available models
        available_models = self.get_all_available_models()
        
        if options is None or len(options) == 0:
            # Task categorization and model selection logic
            task_lower = task_description.lower()
            
            # SQL related tasks - use SQLCoder if available
            if any(keyword in task_lower for keyword in ["sql", "database", "query", "table"]):
                if "sqlcoder:15b" in available_models:
                    self._logger.info("Selected sqlcoder:15b for SQL-related task")
                    return "sqlcoder:15b"
            
            # Code generation tasks - use specialized coding models
            if any(keyword in task_lower for keyword in ["code", "programming", "function", "class", "implementation"]):
                if "qwen2.5-coder:14b" in available_models:
                    self._logger.info("Selected qwen2.5-coder:14b for code generation task")
                    return "qwen2.5-coder:14b"
                elif "opencoder:8b" in available_models:
                    self._logger.info("Selected opencoder:8b for code generation task")
                    return "opencoder:8b"
            
            # Complex reasoning tasks - use larger models
            if any(keyword in task_lower for keyword in ["complex", "reasoning", "difficult", "advanced", "analyze", "explain"]):
                if "deepseek-r1:70b" in available_models:
                    self._logger.info("Selected deepseek-r1:70b for complex reasoning task")
                    return "deepseek-r1:70b"
                elif "llama3.3" in available_models:
                    self._logger.info("Selected llama3.3 for complex reasoning task")
                    return "llama3.3"
            
            # Report generation or text formatting
            if any(keyword in task_lower for keyword in ["report", "format", "summarize", "summarization"]):
                if "deepseek-r1:14b" in available_models:
                    self._logger.info("Selected deepseek-r1:14b for report generation task")
                    return "deepseek-r1:14b"
            
            # Default to configured models based on environment
            self._logger.info(f"Using default model: {self.default_model}")
            return self.default_model
        else:
            # Choose from provided options based on task
            task_lower = task_description.lower()
            
            # SQL tasks
            if any(keyword in task_lower for keyword in ["sql", "database", "query"]):
                for model in ["sqlcoder:15b", "deepseek-r1:14b", "qwen2.5-coder:14b"]:
                    if model in options:
                        self._logger.info(f"Selected {model} from options for SQL task")
                        return model
            
            # Code generation tasks
            if any(keyword in task_lower for keyword in ["code", "programming", "implementation"]):
                for model in ["qwen2.5-coder:14b", "opencoder:8b", "deepseek-r1:14b"]:
                    if model in options:
                        self._logger.info(f"Selected {model} from options for coding task")
                        return model
            
            # Complex reasoning
            if any(keyword in task_lower for keyword in ["complex", "reasoning", "difficult"]):
                for model in ["deepseek-r1:70b", "llama3.3", "deepseek-r1:14b"]:
                    if model in options:
                        self._logger.info(f"Selected {model} from options for reasoning task")
                        return model
                        
            # Default to first option if no specific match
            self._logger.info(f"Selected first option model: {options[0]}")
            return options[0]


class EnhancedOllamaLLM(LLM):
    """Enhanced Ollama LLM with timing and logging."""

    model: str
    base_url: str
    temperature: float = 0.0  # Changed default to 0 for deterministic output
    top_p: float = 0.9
    num_ctx: int = 16384  # Increased context size for larger prompts
    num_thread: int = 12  # Increased thread count for better parallelization
    num_gpu: int = 1      # Number of GPUs to use
    num_batch: int = 512  # Batch size for processing
    f16: bool = True      # Use FP16 for better performance when available
    rope_scaling: float = 1.0  # RoPE frequency scaling
    mlock: bool = True    # Lock model in memory for faster inference
    
    # Pydantic configuration
    model_config = ConfigDict(extra='allow')
    
    _ollama: CrewAICompatibleOllamaLLM = None
    _logger = None

    def __init__(self, **kwargs):
        """Initialize the EnhancedOllamaLLM."""
        super().__init__(**kwargs)
        self._logger = AgentLogger(
            agent_name=f"OllamaLLM-{self.model}", agent_type="LLM"
        )
        
        # Get configuration from environment
        num_ctx = int(getattr(config, "OLLAMA_NUM_CTX", 16384)) if hasattr(config, "OLLAMA_NUM_CTX") else 16384
        num_thread = int(getattr(config, "OLLAMA_NUM_THREAD", 12)) if hasattr(config, "OLLAMA_NUM_THREAD") else 12
        num_gpu = int(getattr(config, "OLLAMA_NUM_GPU", 1)) if hasattr(config, "OLLAMA_NUM_GPU") else 1
        num_batch = int(getattr(config, "OLLAMA_NUM_BATCH", 512)) if hasattr(config, "OLLAMA_NUM_BATCH") else 512
        f16 = getattr(config, "OLLAMA_F16", "true").lower() == "true" if hasattr(config, "OLLAMA_F16") else True
        rope_scaling = float(getattr(config, "OLLAMA_ROPE_SCALING", 1.0)) if hasattr(config, "OLLAMA_ROPE_SCALING") else 1.0
        mlock = getattr(config, "OLLAMA_MLOCK", "true").lower() == "true" if hasattr(config, "OLLAMA_MLOCK") else True
        temperature = float(getattr(config, "OLLAMA_TEMPERATURE", 0.0)) if hasattr(config, "OLLAMA_TEMPERATURE") else 0.0
        
        self._logger.info(f"Initializing {self.model} with optimized parameters:")
        self._logger.info(f"  - Temperature: {temperature}")
        self._logger.info(f"  - Context size: {num_ctx}")
        self._logger.info(f"  - Threads: {num_thread}")
        self._logger.info(f"  - GPUs: {num_gpu}")
        self._logger.info(f"  - Batch size: {num_batch}")
        self._logger.info(f"  - FP16: {f16}")
        self._logger.info(f"  - RoPE scaling: {rope_scaling}")
        self._logger.info(f"  - Memory lock: {mlock}")
        
        # Initialize the CrewAI-compatible OllamaLLM with optimized parameters
        self._ollama = CrewAICompatibleOllamaLLM(
            model=self.model,
            base_url=self.base_url,
            temperature=temperature,
            top_p=self.top_p,
            num_ctx=num_ctx,
            num_thread=num_thread,
            num_gpu=num_gpu,
            num_batch=num_batch,
            f16=f16,
            rope_scaling=rope_scaling,
            mlock=mlock,
        )
        
    @property
    def _llm_type(self) -> str:
        """Return the LLM type.

        Returns:
            str: LLM type
        """
        return "enhanced_ollama"
    
    # Add CrewAI compatibility methods
    def supports_stop_words(self) -> bool:
        """Method required by CrewAI for stop token handling.
        
        Returns:
            bool: Whether this LLM supports stop words
        """
        return True
    
    def get_num_tokens(self, text: str) -> int:
        """Method for counting tokens in text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: Approximate token count
        """
        return self._ollama.get_num_tokens(text)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Call the LLM with the given prompt.

        Args:
            prompt: Input prompt
            stop: Optional stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments

        Returns:
            str: Generated text
        """
        self._logger.info(f"Generating response with model {self.model}")
        
        # Log the prompt (truncated for readability)
        prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
        self._logger.info(f"PROMPT: {prompt_preview}")
        
        start_time = time.time()
        
        # Use the CrewAI-compatible OllamaLLM
        response = self._ollama.invoke(prompt, stop=stop, **kwargs)
        
        generation_time = time.time() - start_time
        self._logger.info(f"Response generated in {generation_time:.2f} seconds")
        
        # Log the response (truncated for readability)
        response_preview = response[:500] + "..." if len(response) > 500 else response
        self._logger.info(f"RESPONSE: {response_preview}")
        
        return response

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        """Stream the LLM response.

        Args:
            prompt: Input prompt
            stop: Optional stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments

        Returns:
            Iterator of text chunks
        """
        self._logger.info(f"Streaming response with model {self.model}")
        start_time = time.time()
        
        # Use the CrewAI-compatible OllamaLLM
        for chunk in self._ollama.stream(prompt, stop=stop, **kwargs):
            if run_manager:
                run_manager.on_llm_new_token(chunk)
            yield chunk
        
        streaming_time = time.time() - start_time
        self._logger.info(f"Streaming completed in {streaming_time:.2f} seconds")


def get_ollama_llm(model_name: str = None) -> EnhancedOllamaLLM:
    """Get an Ollama LLM instance.

    Args:
        model_name: Name of the model to use

    Returns:
        EnhancedOllamaLLM: The LLM instance
    """
    manager = OllamaManager()
    return manager.get_model(model_name) 