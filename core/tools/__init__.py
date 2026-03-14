# ==== core/tools/__init__.py ====
"""
Tool Engine for the Personal LLM Orchestrator.

Exposes a dictionary mapping tool names to their actual Python 
implementations for easy execution by the orchestrator.
"""

from core.tools.api_tools import get_current_time, get_weather
from core.tools.os_tools import open_folder, open_file

# Map tool names to their corresponding python methods
TOOL_REGISTRY = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "open_folder": open_folder,
    "open_file": open_file,
}

# Provide the JSON Schema definitions for the available tools.
# This schema aligns with the standard OpenAI and Ollama Function Calling specs.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the exact current date, time, and timezone.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather forecast for a specific city or geographical location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state/country, e.g., 'San Francisco, CA' or 'London'"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_folder",
            "description": "Open a folder or directory on the host operating system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_path": {
                        "type": "string",
                        "description": "The absolute or relative path to the folder to open."
                    }
                },
                "required": ["folder_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_file",
            "description": "Open a specific file on the host operating system using its default application.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path to the file to open."
                    }
                },
                "required": ["file_path"]
            }
        }
    }
]

__all__ = ["TOOL_REGISTRY", "TOOL_DEFINITIONS", "get_current_time", "get_weather", "open_folder", "open_file"]
