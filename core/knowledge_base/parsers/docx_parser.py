# ==== core/knowledge_base/parsers/docx_parser.py ====
"""
Word Document Parser.

Extracts text from .docx files using python-docx.
"""

from pathlib import Path
from core.utils.logger import get_logger

logger = get_logger(__name__)

def parse_docx(file_path: Path) -> str | None:
    """Extract all text from a Word document (.docx).

    Args:
        file_path: Path to the .docx file.

    Returns:
        Extracted text as a single string, or None if extraction fails.
    """
    try:
        import docx
        
        doc = docx.Document(file_path)
        text_blocks = [para.text for para in doc.paragraphs if para.text]
        return "\n\n".join(text_blocks)
            
    except ImportError:
        logger.error("python-docx is not installed. Run `pip install python-docx`.")
        return None
    except Exception as exc:
        logger.error("Failed to parse DOCX '%s': %s", file_path.name, exc)
        return None
