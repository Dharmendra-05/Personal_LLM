# ==== models/api_keys/anthropic_key.py ====
"""
Anthropic API Key Configuration.

Loads and validates the ANTHROPIC_API_KEY from the environment.
"""

from pydantic_settings import BaseSettings


class AnthropicKeySettings(BaseSettings):
    """Configuration for Anthropic API Key."""
    
    api_key: str | None = None
    
    class Config:
        env_prefix = "ANTHROPIC_"

def get_anthropic_settings() -> AnthropicKeySettings:
    return AnthropicKeySettings()
