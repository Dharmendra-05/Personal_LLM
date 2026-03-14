# ==== core/knowledge_base/parsers/pdf_parser.py ====
"""
PDF Document Parser.

Extracts text from PDF files using PyPDF2.
"""

from pathlib import Path
from core.utils.logger import get_logger

logger = get_logger(__name__)

def parse_pdf(file_path: Path) -> str | None:
    """Extract all text from a PDF file.

    Args:
        file_path: Path to the .pdf file.

    Returns:
        Extracted text as a single string, or None if extraction fails.
    """
    try:
        import PyPDF2
        
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text_blocks = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_blocks.append(text)
            return "\n\n".join(text_blocks)
            
    except ImportError:
        logger.error("PyPDF2 is not installed. Run `pip install pypdf2`.")
        return None
    except Exception as exc:
        logger.error("Failed to parse PDF '%s': %s", file_path.name, exc)
        return None
