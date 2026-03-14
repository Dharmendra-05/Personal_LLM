# ==== models/api_keys/__init__.py ====
"""
API Keys Stitcher.

Imports all individual key configurations (like tributaries) and aggregates them
into a single, unified `APIKeyManager` (the river).
"""

from typing import Any
from pydantic import BaseModel

from models.api_keys.openai_key import get_openai_settings
from models.api_keys.anthropic_key import get_anthropic_settings
from models.api_keys.google_key import get_google_settings
from models.api_keys.groq_key import get_groq_settings
from models.api_keys.hf_key import get_hf_settings


class APIKeyManager(BaseModel):
    """Unified API Key Manager holding all provider configurations."""
    
    openai: str | None = None
    anthropic: str | None = None
    google: str | None = None
    groq: str | None = None
    hf: str | None = None
    
    @classmethod
    def load_all_keys(cls) -> "APIKeyManager":
        """Load all keys from their respective individual configurations."""
        openai_settings = get_openai_settings()
        anthropic_settings = get_anthropic_settings()
        google_settings = get_google_settings()
        groq_settings = get_groq_settings()
        hf_settings = get_hf_settings()
        
        return cls(
            openai=openai_settings.api_key,
            anthropic=anthropic_settings.api_key,
            google=google_settings.api_key,
            groq=groq_settings.api_key,
            hf=hf_settings.api_key,
        )

# A singleton instance of the unified key manager
provider_keys = APIKeyManager.load_all_keys()

__all__ = ["APIKeyManager", "provider_keys"]
