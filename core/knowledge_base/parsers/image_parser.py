# ==== core/knowledge_base/parsers/image_parser.py ====
"""
Image Document Parser.

Extracts a text description from images using a local Ollama Vision model (e.g., llava).
"""

import base64
from pathlib import Path
import requests
from core.utils.logger import get_logger

logger = get_logger(__name__)

def parse_image(file_path: Path, model_name: str = "llava:latest", base_url: str = "http://localhost:11434") -> str | None:
    """Extract a text description from an image file using Ollama.

    Args:
        file_path: Path to the image file (.png, .jpg, .jpeg).
        model_name: The vision model to use (default: 'llava:latest').
        base_url: Ollama server base URL.

    Returns:
        Extracted text description, or None if extraction fails.
    """
    try:
        # Read and encode the image in base64
        with open(file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = "Describe this image in detail. Extract any text written in the image explicitly."

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "images": [base64_image]
        }
        
        url = f"{base_url.rstrip('/')}/api/generate"
        logger.debug("parse_image() → POST %s | model=%s", url, model_name)
        
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 404:
            logger.error(
                "Vision model '%s' not found. Run `ollama run %s` in a separate terminal first.", 
                model_name, model_name
            )
            return None
            
        if not response.ok:
            logger.error("parse_image() failed with HTTP %d: %s", response.status_code, response.text)
            return None
            
        data = response.json()
        
        description = data.get("response", "").strip()
        if description:
            return f"--- Image: {file_path.name} ---\n{description}"
            
        return None

    except ConnectionError:
        logger.error("Failed to connect to Ollama at %s for image parsing.", base_url)
        return None
    except Exception as exc:
        logger.error("Failed to parse Image '%s': %s", file_path.name, exc)
        return None
