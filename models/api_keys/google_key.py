# ==== models/api_keys/google_key.py ====
"""
Google (Gemini) API Key Configuration.

Loads and validates the GOOGLE_API_KEY from the environment.
"""

from pydantic_settings import BaseSettings


class GoogleKeySettings(BaseSettings):
    """Configuration for Google API Key."""
    
    api_key: str | None = None
    
    class Config:
        env_prefix = "GOOGLE_"

def get_google_settings() -> GoogleKeySettings:
    return GoogleKeySettings()
