# ==== core/knowledge_base/parsers/__init__.py ====
"""
Modular document parsers for ingesting multimodal file formats.
"""

from core.knowledge_base.parsers.pdf_parser import parse_pdf
from core.knowledge_base.parsers.docx_parser import parse_docx
from core.knowledge_base.parsers.xlsx_parser import parse_xlsx
from core.knowledge_base.parsers.image_parser import parse_image
from core.knowledge_base.parsers.audio_parser import parse_audio

__all__ = ["parse_pdf", "parse_docx", "parse_xlsx", "parse_image", "parse_audio"]
