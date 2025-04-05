from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from langchain.vectorstores.base import VectorStore as LangchainVectorStore

class VectorStore(ABC):
    """Abstract base class for vector store implementations"""
    
    @abstractmethod
    def create_and_save(self, metadata_info: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Create and save vector stores
        
        Args:
            metadata_info: Optional metadata information
            
        Returns:
            Dictionary mapping names to collection names
        """
        pass
    
    @abstractmethod
    def load(self) -> LangchainVectorStore:
        """
        Load vector store
        
        Returns:
            Loaded vector store
        """
        pass