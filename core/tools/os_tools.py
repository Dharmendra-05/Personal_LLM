# ==== core/tools/os_tools.py ====
"""
Agentic Tools: Operating System Interactions

Functions that allow the LLM to interact with the local machine.
"""

import os
import sys
import subprocess
from pathlib import Path
from core.utils.logger import get_logger

logger = get_logger(__name__)

def open_folder(folder_path: str) -> str:
    """Open a folder on the host operating system.

    Args:
        folder_path: The directory path to open.

    Returns:
        Status message of the action.
    """
    path = Path(folder_path).resolve()
    if not path.exists():
        return f"Directory '{path}' does not exist on this machine."
    if not path.is_dir():
        return f"'{path}' is a file, not a directory."
        
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", str(path)], check=True)
        else:  # Linux (including WSL if configured correctly for xdg-open)
            # In WSL, explorer.exe might be preferred to open Windows folders
            if "microsoft-standard" in os.uname().release.lower():
                subprocess.run(["explorer.exe", str(path)], check=True)
            else:
                subprocess.run(["xdg-open", str(path)], check=True)
                
        return f"Successfully opened the folder: {path}"
    except Exception as exc:
        logger.error("Failed to open folder %s: %s", path, exc)
        return f"Encountered an error trying to open the folder: {exc}"

def open_file(file_path: str) -> str:
    """Open a specific file using the host operating system's default application.

    Args:
        file_path: The file path to open.

    Returns:
        Status message of the action.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return f"File '{path}' does not exist on this machine."
    if not path.is_file():
        return f"'{path}' is a directory, not a file."
        
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=True)
        else:
            if "microsoft-standard" in os.uname().release.lower():
                subprocess.run(["explorer.exe", str(path)], check=True)
            else:
                subprocess.run(["xdg-open", str(path)], check=True)
                
        return f"Successfully opened the file: {path}"
    except Exception as exc:
        logger.error("Failed to open file %s: %s", path, exc)
        return f"Encountered an error trying to open the file: {exc}"
