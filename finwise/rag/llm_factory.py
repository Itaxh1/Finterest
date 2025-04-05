from typing import Optional
from langchain.llms.base import LLM
from langchain_groq import ChatGroq
from config.config import Config, LLMProvider
from config.logger import Logger
import traceback as tb
import os

class LLMFactory:
    """Factory class for creating different LLM instances"""
    
    @staticmethod
    def create_llm(config: Config) -> Optional[LLM]:
        """
        Create an LLM instance based on the config
        
        Args:
            config: The configuration object
            
        Returns:
            An instance of the specified LLM
        """
        logger = Logger().get_logger()
        try:
            if config.llm_provider == LLMProvider.GROQ:
                # Ensure API key is set
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    logger.error("GROQ_API_KEY environment variable not set")
                    raise ValueError("GROQ_API_KEY environment variable not set")
                
                logger.info(f"Creating Groq LLM with model: {config.groq_model_name}")
                return ChatGroq(
                    model_name=config.groq_model_name,
                    temperature=config.llm_params.get('temperature', 0.1),
                    top_p=config.llm_params.get('top_p', 0.9)
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
        except Exception as e:
            logger.error(f"Error creating LLM: {str(e)}\n{tb.format_exc()}")
            return None