from typing import List, Dict, Any, Optional
from langchain.llms.base import LLM
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from config.config import Config
from config.logger import Logger
import traceback as tb
from langsmith import traceable

class QueriesOutput(BaseModel):
    """Schema for the generated queries"""
    queries: List[str] = Field(description="List of search queries")

class QueryGenerator:
    """Class for generating multiple queries from a single user query"""
    
    def __init__(self, 
                 config: Config, 
                 llm: Optional[LLM] = None, 
                 retriever: Optional[BaseRetriever] = None):
        """
        Initialize QueryGenerator
        
        Args:
            config: Configuration object
            llm: Optional LLM instance
            retriever: Optional retriever instance
        """
        self.logger = Logger().get_logger()
        self.config = config
        self.llm = llm
        self.retriever = retriever
        self.num_queries = config.num_queries
    
    @traceable(name="generate_queries")
    def generate_queries(self, question: str) -> List[str]:
        """
        Generate multiple search queries from a single user question
        
        Args:
            question: User question
            
        Returns:
            List of search queries
        """
        try:
            if not self.llm:
                self.logger.error("LLM not initialized")
                return [question]  # Return original question if LLM not available
            
            self.logger.info(f"Generating {self.num_queries} search queries from: {question}")
            
            # Use an output parser to extract queries
            parser = JsonOutputParser(pydantic_object=QueriesOutput)
            
            # Template for generating search queries
            template = """You are an expert in finance. Your task is to convert a user question into {num_queries} different search queries 
            that will help retrieve relevant information about financial concepts, strategies, and education.
            
            Original question: {question}
            
            Generate {num_queries} different search queries that would help find comprehensive information to answer this question.
            Focus on financial terms, concepts, and educational content. The queries should be diverse and cover different aspects of the question.
            
            {format_instructions}
            """
            
            # Create prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=["question", "num_queries"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            # Create chain
            chain = prompt | self.llm | parser
            
            # Generate queries
            result = chain.invoke({
                "question": question,
                "num_queries": self.num_queries
            })
            
            queries = result.queries
            self.logger.info(f"Generated queries: {queries}")
            
            return queries
            
        except Exception as e:
            self.logger.error(f"Error generating queries: {str(e)}\n{tb.format_exc()}")
            return [question]  # Return original question on error
    
    @traceable(name="retrieve_with_multiple_queries")
    def retrieve_with_multiple_queries(self, question: str) -> List[Document]:
        """
        Retrieve documents using multiple queries generated from the question
        
        Args:
            question: User question
            
        Returns:
            List of retrieved documents
        """
        try:
            if not self.retriever:
                self.logger.error("Retriever not initialized")
                return []
            
            # Generate queries
            queries = self.generate_queries(question)
            
            # Retrieve documents for each query
            all_docs = []
            seen_sources = set()
            
            for query in queries:
                docs = self.retriever.get_relevant_documents(query)
                
                # Remove duplicates based on source + page
                for doc in docs:
                    source = doc.metadata.get('source', '')
                    page = doc.metadata.get('page', 0)
                    doc_id = f"{source}_{page}"
                    
                    if doc_id not in seen_sources:
                        all_docs.append(doc)
                        seen_sources.add(doc_id)
            
            self.logger.info(f"Retrieved {len(all_docs)} unique documents from {len(queries)} queries")
            
            return all_docs
            
        except Exception as e:
            self.logger.error(f"Error retrieving with multiple queries: {str(e)}\n{tb.format_exc()}")
            
            # Fall back to original retriever if multi-query fails
            try:
                if self.retriever:
                    self.logger.info("Falling back to original retriever")
                    return self.retriever.get_relevant_documents(question)
            except Exception:
                pass
                
            return []