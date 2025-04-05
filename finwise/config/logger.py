import logging
import os
from typing import Optional

class Logger:
    """Singleton Logger class"""
    _instance: Optional[logging.Logger] = None
    
    def get_logger(self) -> logging.Logger:
        """
        Get the logger instance
        
        Returns:
            logging.Logger: The logger instance
        """
        if Logger._instance is None:
            # Create logger
            logger = logging.getLogger('finance_rag')
            
            # Set level from environment or default to INFO
            log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
            logger.setLevel(getattr(logging, log_level, logging.INFO))
            
            # Create console handler and set level
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level, logging.INFO))
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Add formatter to handler
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(console_handler)
            
            # Store the logger
            Logger._instance = logger
        
        return Logger._instance