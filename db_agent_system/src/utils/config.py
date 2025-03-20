"""
Configuration module for the DB Agent System.
Loads environment variables and provides configuration settings for the application.
"""
import os
from typing import Dict, Optional, Any
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the DB Agent System."""

    # Database configuration
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_USER: str = os.getenv("DB_USER", "dbuser")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "dbpassword")
    DB_NAME: str = os.getenv("DB_NAME", "sample_db")
    DB_URI: str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Ollama configuration
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "localhost")
    OLLAMA_PORT: int = int(os.getenv("OLLAMA_PORT", "11434"))
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")
    OLLAMA_DEFAULT_MODEL: str = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3")
    OLLAMA_SMALL_MODEL: str = os.getenv("OLLAMA_SMALL_MODEL", "tinyllama")
    OLLAMA_LARGE_MODEL: str = os.getenv("OLLAMA_LARGE_MODEL", "mixtral")

    # Application settings
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # LLM API Keys (optional, for fallback)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    @classmethod
    def load_env(cls, env_file: str) -> None:
        """Load environment variables from a specific .env file.
        
        Args:
            env_file: Path to the .env file
        """
        env_path = Path(env_file)
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        
        # Load environment variables from the specified file
        load_dotenv(env_file, override=True)
        
        # Update the database URI
        cls.DB_HOST = os.getenv("DB_HOST", cls.DB_HOST)
        cls.DB_PORT = int(os.getenv("DB_PORT", str(cls.DB_PORT)))
        cls.DB_USER = os.getenv("DB_USER", cls.DB_USER)
        cls.DB_PASSWORD = os.getenv("DB_PASSWORD", cls.DB_PASSWORD)
        cls.DB_NAME = os.getenv("DB_NAME", cls.DB_NAME)
        cls.DB_URI = f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
        
        # Update Ollama configuration
        cls.OLLAMA_HOST = os.getenv("OLLAMA_HOST", cls.OLLAMA_HOST)
        cls.OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", str(cls.OLLAMA_PORT)))
        cls.OLLAMA_URL = os.getenv("OLLAMA_URL", f"http://{cls.OLLAMA_HOST}:{cls.OLLAMA_PORT}")
        cls.OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", cls.OLLAMA_DEFAULT_MODEL)
        
        # Update other settings
        cls.LOG_LEVEL = os.getenv("LOG_LEVEL", cls.LOG_LEVEL)

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Return the configuration as a dictionary.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and not callable(value)
        }

    @classmethod
    def validate(cls) -> bool:
        """Validate the configuration.

        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        # Basic validation
        if not cls.DB_HOST or not cls.DB_NAME:
            return False
        if not cls.OLLAMA_URL:
            return False
        return True


# Create a singleton instance
config = Config()

# Validate configuration
if not config.validate():
    raise ValueError("Invalid configuration. Please check your environment variables.") 