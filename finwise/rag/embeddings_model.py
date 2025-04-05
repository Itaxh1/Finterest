from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional
from config.config import Config
from config.logger import Logger
import traceback as tb
import streamlit as st

class EmbeddingsModel:
    """Singleton class for managing embeddings models"""
    _instance = None
    _model_name = None
    
    @classmethod
    def get_instance(cls, embedding_client: str = "huggingface", model_name: str = None):
        """
        Returns a singleton embedding model instance.
        
        Args:
            embedding_client (str): "huggingface" (only supported client for now)
            model_name (str): name of the embedding model
        
        Returns:
            Embeddings object
        """
        # Use model name from config if not specified
        if model_name is None:
            model_name = Config().embedding_model
        
        # Return existing instance if it matches the requested model
        if cls._model_name == model_name and cls._instance is not None:
            return cls._instance
            
        logger = Logger().get_logger()
        
        try:
            if embedding_client == "huggingface":
                logger.info(f"Initializing HuggingFace embedding model: {model_name}")
                
                # Determine device based on availability
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"Using device: {device} for embeddings")
                except ImportError:
                    device = "cpu"
                    logger.warning("PyTorch not available, defaulting to CPU for embeddings")
                
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"trust_remote_code": True, "device": device}
                )
                
            else:
                raise ValueError(f"Unsupported embedding client: {embedding_client}")
            
            # Cache in singleton + optionally Streamlit session state
            cls._instance = embeddings
            cls._model_name = model_name
            
            # Store in Streamlit session state if available
            if 'st' in globals():
                if "embedding_model" not in st.session_state:
                    logger.info(f"Initializing embedding model in session: {model_name}")
                else:
                    logger.info(f"Replacing existing embedding model with: {model_name}")
                
                st.session_state.embedding_model_name = model_name
                st.session_state.embedding_model = embeddings
            
            return embeddings
            
        except Exception:
            logger.error(f"Failed to initialize embedding model: {tb.format_exc()}")
            return None