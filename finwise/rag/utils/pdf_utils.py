import os
from typing import List, Optional
from config.config import Config
from config.logger import Logger
import traceback as tb

def get_pdf_files(directory: Optional[str] = None) -> List[str]:
    """
    Get all PDF files in the specified directory or the configured directory
    
    Args:
        directory: Directory path (optional, uses config if not provided)
        
    Returns:
        List of PDF file paths
    """
    logger = Logger().get_logger()
    
    try:
        # Use provided directory or get from config
        if directory is None:
            config = Config()
            directory = config.PDF_DIRECTORY
        
        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return []
        
        pdf_files = []
        
        # Walk through the directory and collect PDF files
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        return pdf_files
        
    except Exception as e:
        logger.error(f"Error getting PDF files: {str(e)}\n{tb.format_exc()}")
        return []

def extract_title_from_pdf(pdf_path: str) -> str:
    """
    Extract title from PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted title or filename
    """
    logger = Logger().get_logger()
    
    try:
        # Try to extract title from PDF metadata
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        
        # Close the document
        doc.close()
        
        # Check if title exists in metadata
        if metadata and 'title' in metadata and metadata['title']:
            return metadata['title']
        
        # If no title, use the filename without extension
        return os.path.splitext(os.path.basename(pdf_path))[0]
        
    except Exception as e:
        logger.error(f"Error extracting title from PDF: {str(e)}\n{tb.format_exc()}")
        # Fall back to filename
        return os.path.splitext(os.path.basename(pdf_path))[0]