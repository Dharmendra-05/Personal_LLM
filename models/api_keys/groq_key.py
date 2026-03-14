# ==== models/api_keys/groq_key.py ====
"""
Groq API Key Configuration.

Loads and validates the GROQ_API_KEY from the environment.
"""

from pydantic_settings import BaseSettings


class GroqKeySettings(BaseSettings):
    """Configuration for Groq API Key."""
    
    api_key: str | None = None
    
    class Config:
        env_prefix = "GROQ_"

def get_groq_settings() -> GroqKeySettings:
    return GroqKeySettings()
