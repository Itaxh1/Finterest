from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

class DocumentProcessor(ABC):
    """Abstract base class for document processors"""
    
    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """
        Load documents from a source
        
        Args:
            source: Path to the documents
            
        Returns:
            List of documents
        """
        pass
    
    @abstractmethod
    def split(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        pass
    
    @abstractmethod
    def process_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Process metadata for documents
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of documents with processed metadata
        """
        pass