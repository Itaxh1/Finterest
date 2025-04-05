from typing import List, Dict, Any, Callable
from langchain.schema import Document
from config.logger import Logger
import os
import traceback as tb

def process_documents(documents: List[Document], 
                      metadata_processor: Callable, 
                      **kwargs) -> List[Document]:
    """
    Process documents with the specified metadata processor
    
    Args:
        documents: List of documents to process
        metadata_processor: Function to process document metadata
        **kwargs: Additional arguments to pass to the metadata processor
        
    Returns:
        List of processed documents
    """
    logger = Logger().get_logger()
    processed_docs = []
    
    for doc in documents:
        try:
            processed_metadata = metadata_processor(doc.metadata, **kwargs)
            processed_doc = Document(
                page_content=doc.page_content,
                metadata=processed_metadata
            )
            processed_docs.append(processed_doc)
        except Exception as e:
            logger.error(f"Error processing document metadata: {str(e)}\n{tb.format_exc()}")
            # Add original document if processing fails
            processed_docs.append(doc)
    
    return processed_docs

def process_metadata_for_finance(metadata: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Process metadata for finance documents
    
    Args:
        metadata: Original document metadata
        **kwargs: Additional parameters
        
    Returns:
        Processed metadata
    """
    processed_metadata = metadata.copy()
    
    # Extract filename and title
    source_path = metadata.get('source', '')
    filename = os.path.basename(source_path)
    
    # Add finance-specific metadata
    processed_metadata.update({
        'domain': 'finance',
        'filename': filename,
        'content_type': 'finance_education'
    })
    
    # Add book title if available
    title = kwargs.get('title')
    if title:
        processed_metadata['book_title'] = title
    
    # Add any additional metadata provided in kwargs
    for key, value in kwargs.items():
        if key != 'title' and value:  # Skip title as it's already handled
            processed_metadata[key] = value
    
    return processed_metadata

def process_metadata_for_combined(metadata: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Process metadata for combined collection
    
    Args:
        metadata: Original document metadata
        **kwargs: Additional parameters
        
    Returns:
        Processed metadata
    """
    processed_metadata = metadata.copy()
    
    # Extract filename and basic info
    source_path = metadata.get('source', '')
    filename = os.path.basename(source_path)
    
    # Add general metadata
    processed_metadata.update({
        'domain': 'finance',
        'filename': filename,
        'content_type': 'finance_education',
        'collection_type': 'combined'
    })
    
    return processed_metadata