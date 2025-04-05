Finance Education RAG Chatbot
This project implements a Retrieval-Augmented Generation (RAG) chatbot focused on finance education. The chatbot uses ChromaDB as a vector store, BGE-M3 embeddings, and Llama 3 via Groq API for generation.
Features

Loads and processes finance books/PDFs
Creates embeddings using BAAI/bge-m3 model
Uses reranking to improve retrieval quality
Supports multi-query retrieval for better search results
Implements a Streamlit web interface for easy interaction

Project Structure

finance_rag/
├── config/               # Configuration and logging
├── data/                 # Store finance PDFs here
│   └── finance_books/
├── chroma_vector_store/  # Vector database (created at runtime)
├── rag/                  # RAG implementation
│   ├── document_processor/
│   ├── embeddings_model.py
│   ├── llm_factory.py
│   ├── metadata/
│   ├── query_generator.py
│   ├── rag_pipeline.py
│   ├── utils/
│   └── vector_store/
├── app.py               # Streamlit web application
├── requirements.txt
└── .env                 # Environment variables


to create env - conda create -n finance_rag python=3.10.15

to activate - conda activate finance_rag

to install requirements - pip install -r requirements.txt