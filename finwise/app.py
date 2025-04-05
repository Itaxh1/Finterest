import streamlit as st
import os
from datetime import datetime
from config.config import Config
from config.logger import Logger
from rag.rag_pipeline import RAGPipeline
from rag.vector_store.chroma_vector_store import ChromaVectorStore
import traceback as tb

# Configure page
st.set_page_config(
    page_title="Finance Education Chatbot",
    page_icon="💰",
    layout="wide"
)

# Initialize logger
logger = Logger().get_logger()

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_initialized = False
        
    if "pipeline_needs_restart" not in st.session_state:
        st.session_state.pipeline_needs_restart = False

def create_or_reload_pipeline():
    """Create a new RAG pipeline instance or reload it if needed"""
    config = Config()
    st.session_state.rag_pipeline = RAGPipeline(config)
    success = st.session_state.rag_pipeline.setup()
    st.session_state.pipeline_needs_restart = False
    return success

def check_vector_store():
    """Check if vector store exists and has documents"""
    config = Config()
    chroma_path = config.CHROMA_DIRECTORY
    
    try:
        if os.path.exists(chroma_path) and len(os.listdir(chroma_path)) > 0:
            # Check if we have the finance collection
            try:
                # Create a temporary ChromaVectorStore to check collection counts
                vector_store = ChromaVectorStore(config)
                client = vector_store.client
                
                if "finance" in client.list_collections():
                    collection = client.get_collection("finance")
                    count = collection.count()
                    if count > 0:
                        st.session_state.vector_store_initialized = True
                        return True, count
                    else:
                        return False, 0
                else:
                    return False, 0
            except Exception as e:
                logger.error(f"Error checking vector store: {e}")
                return False, 0
        return False, 0
    except Exception as e:
        logger.error(f"Error checking vector store path: {e}")
        return False, 0

def create_vector_store():
    """Create vector store from PDF documents"""
    try:
        with st.spinner("Creating vector store from finance documents... This may take several minutes."):
            config = Config()
            vector_store = ChromaVectorStore(config)
            collections = vector_store.create_and_save()
            
            if collections:
                st.session_state.vector_store_initialized = True
                # Flag that we need to restart the pipeline to use the new collection
                st.session_state.pipeline_needs_restart = True 
                st.success(f"Vector store created successfully!")
                return True
            else:
                st.error("Failed to create vector store. Check logs for details.")
                return False
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}\n{tb.format_exc()}")
        st.error(f"Error creating vector store: {str(e)}")
        return False

def main():
    # Initialize session state
    initialize_session_state()
    
    # Check if pipeline needs to be restarted
    if "rag_pipeline" not in st.session_state or st.session_state.pipeline_needs_restart:
        create_or_reload_pipeline()
    
    # Page header
    st.title("Finance Education Chatbot 💰")
    st.markdown("""
    Ask me anything about finance concepts, investment strategies, 
    or financial planning. I'm here to help you understand finance better!
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This chatbot uses Retrieval-Augmented Generation (RAG) to provide 
        accurate and educational information about finance topics.
        
        It utilizes:
        - BGE-M3 embeddings from HuggingFace
        - Reranking for improved relevance
        - Llama 3 via Groq for generation
        """)
        
        st.subheader("Vector Store")
        has_vectors, doc_count = check_vector_store()
        
        if has_vectors:
            st.success(f"Vector store loaded with {doc_count} documents!")
            if st.button("Rebuild Vector Store"):
                if create_vector_store():
                    st.experimental_rerun()  # Rerun to refresh the pipeline
        else:
            st.warning("Vector store not found or empty!")
            # Always show the button to create vector store
            if st.button("Create Vector Store"):
                if create_vector_store():
                    st.experimental_rerun()  # Rerun to refresh the pipeline
        
        st.subheader("RAG Settings")
        use_multi_query = st.checkbox("Use Multi-Query Retrieval", value=False)
        
        if st.button("Apply Settings"):
            with st.spinner("Applying settings..."):
                st.session_state.rag_pipeline.setup(use_multi_query=use_multi_query)
                st.success("Settings applied!")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Input for new message
    if prompt := st.chat_input("Ask a question about finance"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    has_vectors, _ = check_vector_store()
                    if not has_vectors:
                        response = "I don't have any finance documents loaded yet. Please create the vector store using the sidebar button first."
                    else:
                        # Ensure pipeline is fresh
                        if st.session_state.pipeline_needs_restart:
                            create_or_reload_pipeline()
                        response = st.session_state.rag_pipeline.query(prompt)
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}\n{tb.format_exc()}")
                    response = "I'm sorry, but I encountered an error while processing your question. Please try again or rephrase your question."
                
                message_placeholder.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()