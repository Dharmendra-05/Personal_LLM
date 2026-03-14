# ==== models/api_keys/openai_key.py ====
"""
OpenAI API Key Configuration.

Loads and validates the OPENAI_API_KEY from the environment.
"""

from pydantic_settings import BaseSettings


class OpenAIKeySettings(BaseSettings):
    """Configuration for OpenAI API Key."""
    
    api_key: str | None = None
    
    class Config:
        env_prefix = "OPENAI_"

def get_openai_settings() -> OpenAIKeySettings:
    return OpenAIKeySettings()
