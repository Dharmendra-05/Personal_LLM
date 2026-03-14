# ==== core/knowledge_base/parsers/xlsx_parser.py ====
"""
Excel Spreadsheet Parser.

Extracts text and table data from .xlsx files using pandas.
"""

from pathlib import Path
from core.utils.logger import get_logger

logger = get_logger(__name__)

def parse_xlsx(file_path: Path) -> str | None:
    """Extract text representation of Excel spreadsheets (.xlsx).

    Args:
        file_path: Path to the .xlsx file.

    Returns:
        Extracted text spanning all sheets, or None if extraction fails.
    """
    try:
        import pandas as pd
        
        sheets = pd.read_excel(file_path, sheet_name=None)
        text_blocks = []
        for sheet_name, df in sheets.items():
            text_blocks.append(f"--- Sheet: {sheet_name} ---")
            # Convert dataframe to string representation
            text_blocks.append(df.to_string(index=False))
            
        return "\n\n".join(text_blocks)
            
    except ImportError:
        logger.error("pandas or openpyxl is not installed. Run `pip install pandas openpyxl`.")
        return None
    except Exception as exc:
        logger.error("Failed to parse XLSX '%s': %s", file_path.name, exc)
        return None
