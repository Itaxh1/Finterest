from .document_processor import DocumentProcessor
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langsmith import traceable
from config.config import Config
from config.logger import Logger
import traceback as tb
import os
import re
from datetime import datetime

class ChromaDocumentProcessor(DocumentProcessor):
    """Document processor implementation for ChromaDB-based vector storage"""
    
    def __init__(self, config: Config, metadata_info: Optional[Dict[str, Any]] = None):
        """
        Initialize ChromaDocumentProcessor
        
        Args:
            config: Configuration object
            metadata_info: Optional dictionary with additional metadata information
        """
        self.logger = Logger().get_logger()
        self.config = config
        self.metadata_info = metadata_info or {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    @traceable(name="load_documents")
    def load(self, source: str) -> List[Document]:
        """
        Load PDF documents from a source (file or directory)
        
        Args:
            source: Path to the PDF file or directory
            
        Returns:
            List of documents
        """
        try:
            self.logger.info(f"Loading documents from: {source}")
            
            if os.path.isdir(source):
                # Load from directory
                loader = PyPDFDirectoryLoader(path=source)
                pdfs = []
                pdfs_lazy = loader.lazy_load()
                
                for pdf in pdfs_lazy:
                    if pdfs and pdfs[-1].metadata['source'] == pdf.metadata['source']:
                        pdfs[-1].page_content += pdf.page_content
                        continue
                    pdfs.append(pdf)
                
                self.logger.info(f"Documents loaded from directory: {len(pdfs)}")
                return pdfs
            
            elif os.path.isfile(source) and source.lower().endswith('.pdf'):
                # Load from single file using PyMuPDF
                from langchain_community.document_loaders import PyMuPDFLoader
                loader = PyMuPDFLoader(file_path=source)
                pdfs = loader.load()
                self.logger.info(f"Document loaded from file: {source}")
                return pdfs
            
            else:
                self.logger.error(f"Source is not a valid PDF file or directory: {source}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error loading documents: {str(e)}\n{tb.format_exc()}")
            return []
    
    @traceable(name="split_documents")
    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        try:
            self.logger.info("Splitting documents into chunks")
            chunks = self.text_splitter.split_documents(documents)
            self.logger.info(f"Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            self.logger.error(f"Error splitting documents: {str(e)}\n{tb.format_exc()}")
            return []
    
    def process_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Process metadata for ChromaDB documents
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of documents with processed metadata
        """
        try:
            processed_docs = []
            
            for doc in documents:
                # Extract filename from source path
                source_path = doc.metadata.get('source', '')
                filename = os.path.basename(source_path)
                
                # Clean and enhance metadata
                enhanced_metadata = {
                    'source': source_path,
                    'filename': filename,
                    'title': self._extract_title(filename),
                    'page': doc.metadata.get('page', 0),
                    'chunk_id': f"{filename}_{doc.metadata.get('page', 0)}_{len(processed_docs)}",
                    'processing_date': datetime.now().isoformat(),
                    'content_type': 'finance_book'
                }
                
                # Add any additional metadata from metadata_info
                if self.metadata_info:
                    enhanced_metadata.update(self.metadata_info)
                
                # Create new document with enhanced metadata
                processed_doc = Document(
                    page_content=doc.page_content,
                    metadata=enhanced_metadata
                )
                
                processed_docs.append(processed_doc)
            
            self.logger.info(f"Processed metadata for {len(processed_docs)} documents")
            return processed_docs
            
        except Exception as e:
            self.logger.error(f"Error processing metadata: {str(e)}\n{tb.format_exc()}")
            return documents
    
    def _extract_title(self, filename: str) -> str:
        """
        Extract a readable title from filename
        
        Args:
            filename: Filename to process
            
        Returns:
            Clean title
        """
        # Remove extension
        title = os.path.splitext(filename)[0]
        
        # Replace underscores and hyphens with spaces
        title = title.replace('_', ' ').replace('-', ' ')
        
        # Remove any non-alphanumeric characters except spaces
        title = re.sub(r'[^\w\s]', '', title)
        
        # Title case
        title = title.title()
        
        return title