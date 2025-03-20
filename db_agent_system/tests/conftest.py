"""
Configuration and shared fixtures for pytest.
"""
import os
import sys
import pytest
from pathlib import Path

from src.utils.config import config
from src.ollama.llm import OllamaManager


def pytest_configure(config):
    """Configure pytest."""
    # Register markers
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "asyncio: mark test as asyncio test")


@pytest.fixture(scope="function", autouse=True)
def disable_auto_mocking():
    """Disable automatic mocking for tests."""
    # This is a no-op fixture but ensures our tests run with real components
    yield 


class BaseTestClass:
    """Base class for all tests that provides common setup/teardown."""

    @classmethod
    def setup_class(cls):
        """Set up the test environment."""
        # Store original environment
        cls.original_env = os.environ.copy()
        
        # Set test environment
        os.environ["ENV"] = "test"
        
        # Load test configuration
        config_path = Path(__file__).parent.parent / ".env.test"
        if config_path.exists():
            config.load_env(str(config_path))
        
        # Create logs directory if it doesn't exist
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

    @classmethod
    def teardown_class(cls):
        """Restore the original environment."""
        for key, value in cls.original_env.items():
            os.environ[key] = value
        
        for key in os.environ.keys() - cls.original_env.keys():
            del os.environ[key]


@pytest.fixture
def ollama_config():
    """Create a test Ollama configuration."""
    return {
        "model_name": "llama3",
        "base_url": "http://localhost:11434",
        "temperature": 0.1,
        "top_p": 0.9,
        "num_ctx": 2048
    }


@pytest.fixture
def check_ollama_available():
    """Skip test if Ollama is not available."""
    ollama_mgr = OllamaManager()
    models = ollama_mgr.get_all_available_models()
    if not models:
        pytest.skip("No Ollama models available - skipping test")
    return models 