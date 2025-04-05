from .vector_store import VectorStore
from ..embeddings_model import EmbeddingsModel
from ..document_processor.chroma_document_processor import ChromaDocumentProcessor
from ..metadata.process_metadata import process_documents, process_metadata_for_finance, process_metadata_for_combined

import os
import re
import datetime
import traceback as tb
import chromadb
from langchain_community.vectorstores import Chroma
from config.config import Config
from config.logger import Logger
from typing import Dict, Any, Optional, List, Union
from langchain.vectorstores.base import VectorStore as LangchainVectorStore
from langchain.schema import Document

class ChromaVectorStore(VectorStore):
    """Implements vector store operations for ChromaDB"""
    
    def __init__(self, config: Optional[Config] = None, metadata_info: Optional[Dict[str, Any]] = None):
        """
        Initialize ChromaVectorStore
        
        Args:
            config: Configuration object (optional)
            metadata_info: Dictionary with additional metadata (optional)
        """
        self.config = config or Config()
        self.logger = Logger().get_logger()
        self.data_logger = Logger().get_logger()  # Using same logger for data processing
        self.chroma_directory = self.config.CHROMA_DIRECTORY
        self.client = None
        self.embeddings = None
        self.metadata_info = metadata_info or {}
        self.document_processor = ChromaDocumentProcessor(self.config, self.metadata_info)
        
        # Initialize ChromaDB client
        self._init_client()
        
    def _init_client(self):
        """Initialize the ChromaDB client"""
        try:
            os.makedirs(self.chroma_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.chroma_directory)
            self.logger.info(f"ChromaDB client initialized at {self.chroma_directory}")
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB client: {str(e)}\n{tb.format_exc()}")
            raise
    
    def create_and_save(self, metadata_info: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Create vector stores for finance documents in ChromaDB.

        Args:
            metadata_info: Dictionary with additional metadata

        Returns:
            Dictionary mapping collection names
        """
        # Update metadata_info if provided
        if metadata_info:
            self.metadata_info.update(metadata_info)
        
        # Set up a persistent directory for ChromaDB
        os.makedirs(self.config.CHROMA_DIRECTORY, exist_ok=True)
        
        collections = {}
        
        # Process PDF files from the finance book directory
        pdf_files = self._get_pdf_files()
        if not pdf_files:
            self.logger.warning("No PDF files found in directory")
            return {}
            
        # Process PDF files
        combined_docs = self._process_pdf_files(pdf_files)
        
        # Initialize embeddings
        embeddings = EmbeddingsModel.get_instance(self.config.embedding_client, self.config.embedding_model)
        if not embeddings:
            self.logger.error("Failed to initialize embeddings model.")
            return {}
        
        # Create combined collection
        try:
            if combined_docs:
                self.logger.info(f"Creating finance collection with {len(combined_docs)} documents")
                
                # Process documents for combined collection
                processed_combined_docs = process_documents(combined_docs, process_metadata_for_combined)
                
                # Collection metadata
                combined_metadata = {
                    "type": "finance",
                    "document_count": len(processed_combined_docs),
                    "creation_time": datetime.datetime.now().isoformat()
                }
                
                # Create collection
                _, name = self._create_chroma_collection(
                    "finance", 
                    processed_combined_docs, 
                    embeddings, 
                    self.client, 
                    combined_metadata
                )
                
                if name:
                    collections["finance"] = name
                
            else:
                self.logger.warning("No documents found for finance collection")
        except Exception as e:
            self.logger.error(f"Error creating finance collection: {str(e)}\n{tb.format_exc()}")
        
        # Log summary
        self.logger.info(f"Created {len(collections)} collections in ChromaDB at {self.config.CHROMA_DIRECTORY}")
        return collections
    
    def _get_pdf_files(self) -> List[str]:
        """
        Get all PDF files in the configured directory
        
        Returns:
            List of PDF file paths
        """
        pdf_dir = self.config.PDF_DIRECTORY
        pdf_files = []
        
        try:
            if not os.path.exists(pdf_dir):
                self.logger.warning(f"PDF directory does not exist: {pdf_dir}")
                return []
                
            for root, _, files in os.walk(pdf_dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
                        
            self.logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
            return pdf_files
            
        except Exception as e:
            self.logger.error(f"Error getting PDF files: {str(e)}\n{tb.format_exc()}")
            return []
    
    def _process_pdf_files(self, pdf_files: List[str]) -> List[Document]:
        """
        Process PDF files into documents
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            List of processed documents
        """
        combined_docs = []
        
        for pdf_path in pdf_files:
            try:
                # Load documents using ChromaDocumentProcessor
                raw_documents = self.document_processor.load(pdf_path)
                
                if not raw_documents:
                    continue
                
                # Process metadata
                processed_documents = self.document_processor.process_metadata(raw_documents)
                
                # Split documents
                docs = self.document_processor.split(processed_documents)
                
                if not docs:
                    continue
                    
                # Add to combined docs
                combined_docs.extend(docs)
                self.data_logger.info(f"Added {len(docs)} chunks from {os.path.basename(pdf_path)} to collection.")
                    
            except Exception as e:
                self.data_logger.error(f"Error processing {pdf_path}: {str(e)}\n{tb.format_exc()}")
                
        return combined_docs
    
    def _create_chroma_collection(self, name: str, docs: List[Document], 
                                 embeddings: Any, client: chromadb.PersistentClient, 
                                 metadata: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Create or update a ChromaDB collection.
        
        Args:
            name: Collection name
            docs: Documents to add
            embeddings: Embeddings model
            client: ChromaDB client
            metadata: Collection metadata
            
        Returns:
            tuple: (collection object, collection name)
        """
        try:
            # Prepare collection metadata
            collection_metadata = metadata or {
                "document_count": len(docs),
                "creation_time": datetime.datetime.now().isoformat()
            }
            
            # First, delete existing collection if it exists
            try:
                client.delete_collection(name=name)
                self.logger.info(f"Deleted existing collection {name} for update")
            except Exception:
                # Collection doesn't exist, which is fine for new creation
                pass
                
            # Create the collection first with metadata
            client.create_collection(
                name=name,
                metadata=collection_metadata
            )
            
            # Now use LangChain's Chroma to add documents
            collection = Chroma(
                collection_name=name,
                embedding_function=embeddings,
                persist_directory=self.config.CHROMA_DIRECTORY,
                client=client
            )
            
            # Add documents to the collection
            collection.add_documents(docs)
            
            self.logger.info(f"Collection created: {name} with {len(docs)} documents")
            return collection, name
            
        except Exception as e:
            self.logger.error(f"Error creating collection {name}: {str(e)}\n{tb.format_exc()}")
            return None, None
    
    def load(self) -> LangchainVectorStore:
        """
        Load existing vector store from ChromaDB
        
        Returns:
            Loaded vector store
        """
        try:
            # Initialize ChromaDB client if not already done
            if not self.client:
                self._init_client()
            
            # Initialize embeddings
            embeddings = EmbeddingsModel.get_instance(self.config.embedding_client, self.config.embedding_model)
            if not embeddings:
                raise ValueError("Failed to initialize embeddings model")
            
            # Check if 'finance' collection exists - ChromaDB v0.6.0+ returns just collection names
            collections = self.client.list_collections()
            
            if "finance" not in collections:
                self.logger.warning("Finance collection not found in ChromaDB. Creating new collection.")
                self.create_and_save()
            
            # Load the vector store
            finance_vector_store = Chroma(
                collection_name="finance",
                embedding_function=embeddings,
                persist_directory=self.config.CHROMA_DIRECTORY,
                client=self.client
            )
            
            # Check collection count - need to use a different approach with v0.6.0+
            try:
                collection = self.client.get_collection("finance")
                count = collection.count()
                self.logger.info(f"Loaded finance vector store with {count} documents")
            except Exception as e:
                self.logger.warning(f"Could not get document count: {str(e)}")
            
            return finance_vector_store
            
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}\n{tb.format_exc()}")
            raise