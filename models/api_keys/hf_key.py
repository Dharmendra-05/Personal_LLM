# ==== models/api_keys/hf_key.py ====
"""
Hugging Face API Key Configuration.

Loads and validates the HF_API_KEY from the environment.
"""

from pydantic_settings import BaseSettings


class HFKeySettings(BaseSettings):
    """Configuration for Hugging Face API Key."""
    
    api_key: str | None = None
    
    class Config:
        env_prefix = "HF_"

def get_hf_settings() -> HFKeySettings:
    return HFKeySettings()
