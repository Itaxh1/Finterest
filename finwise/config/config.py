from enum import Enum
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMProvider(str, Enum):
    """Enum for supported LLM providers"""
    GROQ = "groq"

class Config:
    """Singleton configuration class to store all parameters"""
    _instance: Optional["Config"] = None  # Singleton instance
    
    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize configuration parameters"""
        # Vector store settings
        # Get the directory where config.py is located
        self.config_dir: str = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the absolute path to the vector_store folder
        self.vector_store_path: str = os.path.join(self.config_dir, "..", "vector_store")
        
        # Optionally, normalize the path to remove any redundant separators or up-level references
        self.vector_store_path = os.path.normpath(self.vector_store_path)
        
        # Document processing settings
        self.chunk_size: int = 1500
        self.chunk_overlap: int = 400
        
        # Path to Data Folder 
        self.PDF_DIRECTORY = os.path.join(self.config_dir, "..", "data", "finance_books")
        self.CHROMA_DIRECTORY: str = os.path.join(self.config_dir, "..", "chroma_vector_store")
        self.CHROMA_DIRECTORY = os.path.normpath(self.CHROMA_DIRECTORY)
        
        # Processor type
        self.processor_type = "chroma"
        
        # Embedding model settings
        self.embedding_client = "huggingface"
        
        # LLM settings
        self.llm_provider: LLMProvider = LLMProvider.GROQ
        
        self.groq_model_name: str = "llama3-70b-8192"
        self.groq_models: List[str] = [
            "llama3-70b-8192",
            "llama-3.3-70b-specdec",
            "qwen-2.5-32b",
            "deepseek-r1-distill-llama-70b"
        ]
        
        # Retriever settings
        self.retriever_k: int = 10
        
        # LLM-specific parameters
        self.llm_params: Dict[str, Any] = {
            "temperature": 0.1,
            "top_p": 0.9
        }
        
        # Multi-query setting
        self.num_queries: int = 3
    
    @property
    def embedding_model(self) -> str:
        if self.embedding_client == 'huggingface':
            return "BAAI/bge-m3"
        else:
            raise ValueError(f"Unsupported embedding client: {self.embedding_client}")