from typing import Optional
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langsmith import trace, traceable
from config.config import Config
from config.logger import Logger
from .vector_store.chroma_vector_store import ChromaVectorStore
from .query_generator import QueryGenerator
from .llm_factory import LLMFactory
import traceback as tb

@traceable
class RAGPipeline:
    """Manages the RAG pipeline setup and execution"""
    
    def __init__(self, config: Config):
        """
        Initialize RAG pipeline
        
        Args:
            config: Configuration object
        """
        self.logger = Logger().get_logger()
        self.config = config
        self.vector_store = ChromaVectorStore(config)
        self.query_generator = None
        self.qa_chain = None
        self.llm = None
        self.retriever = None
        self.reranker = None
    
    @traceable(name="setup_pipeline")
    def setup(self, use_multi_query: bool = False) -> bool:
        """ 
        Set up the RAG pipeline
        
        Args:
            use_multi_query: Whether to use multi-query RAG (default: False)
            
        Returns:
            True if setup was successful, False otherwise
        """
        if self.qa_chain is not None:
            self.logger.info("RAG pipeline already set up, reusing existing pipeline")
            return True
            
        try: 
            self.logger.info("Setting up RAG pipeline")
            
            # Create LLM
            self.llm = LLMFactory.create_llm(self.config)
            if not self.llm:
                self.logger.error("Failed to create LLM")
                return False
            
            # Load vector store
            langchain_vector_store = self.vector_store.load()
            
            # Trace the retriever setup
            with trace(name="setup_retriever"):
                # Use only 'k' parameter without 'fetch_k' to avoid errors
                self.retriever = langchain_vector_store.as_retriever(
                    search_kwargs={'k': self.config.retriever_k}
                )

            # Set up query generator
            self.query_generator = QueryGenerator(
                self.config, 
                llm=self.llm, 
                retriever=self.retriever,
            )
            
            # Define system prompt
            system_prompt = SystemMessagePromptTemplate.from_template(
                """You are a knowledgeable finance educator and advisor. You help people understand financial concepts, 
                strategies, and best practices. Your goal is to provide clear, accurate, and helpful information about finance
                to help users improve their financial literacy and make informed decisions.
                
                Always provide balanced and educational information about finance topics. When explaining concepts, use simple language
                and relevant examples. If you're unsure about specific details, acknowledge the limitations of your information.
                
                Do not provide specific investment advice or recommendations for individual situations. Instead, focus on explaining
                general financial principles and educational content."""
            )

            # Define user query format
            user_prompt = HumanMessagePromptTemplate.from_template(
                """Context information from finance resources:
                {context}
                
                User Question: {question}
                
                Answer:"""
            )

            # Combine prompts
            prompt_template = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

            # Set up appropriate retrieval method
            if use_multi_query:
                self.logger.info("Using multi-query retrieval")
                
                # Define a function to retrieve documents using multi-query generation
                @traceable(name="multi_query_retrieve")
                def multi_query_retrieve(question):
                    return self.query_generator.retrieve_with_multiple_queries(question)
                
                # Build QA chain with multi-query retrieval
                with trace(name="setup_qa_chain"):
                    self.qa_chain = (
                        RunnableMap({
                            "context": RunnablePassthrough() | multi_query_retrieve,
                            "question": RunnablePassthrough()
                        })
                        | prompt_template
                        | self.llm
                    )

            else:
                # Set up reranker
                compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
                self.reranker = ContextualCompressionRetriever(
                    base_compressor=compressor, 
                    base_retriever=self.retriever
                )

                self.logger.info("Using standard retrieval with reranking")
                # Trace the prompt template setup
                with trace(name="setup_qa_chain"):
                    self.qa_chain = (
                        RunnableMap({
                            "context": lambda question: self.reranker.get_relevant_documents(question),
                            "question": RunnablePassthrough()
                        })
                        | prompt_template
                        | self.llm
                    )
            
            self.logger.info("RAG pipeline setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up RAG pipeline: {str(e)}\n{tb.format_exc()}")
            return False

    @traceable(name="execute_query")
    def query(self, question: str) -> str:
        """
        Execute a query through the RAG pipeline
        
        Args:
            question: User question
            
        Returns:
            Generated response
        """
        try:
            self.logger.info(f"Processing query: {question}")
            
            if self.qa_chain is None:
                success = self.setup()
                if not success:
                    return "I'm sorry, but I encountered an issue setting up the RAG pipeline. Please try again later."

            # Trace the actual query execution
            with trace(name="process_query"):
                response = self.qa_chain.invoke(question)
                self.logger.info("Query processed successfully")

            return response.content

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}\n{tb.format_exc()}")
            return "I'm sorry, but I encountered an error while processing your question. Please try again or rephrase your question."