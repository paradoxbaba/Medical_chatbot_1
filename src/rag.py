import os
from typing import Dict, Any, Optional, Tuple

from src.helper import (
    load_pdf_files,
    text_split,
    download_embeddings,
    init_pinecone,
    create_or_load_index,
    create_vectorstore
)
from src.config import get_env_var, get_chat_model
from src.prompt import qa_prompt
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def validate_api_keys():
    """Validate required API keys."""
    try:
        get_env_var("PINECONE_API_KEY")
        return True, None
    except ValueError as e:
        return False, str(e)


def setup_pinecone_index(index_name="medical-chatbot-1"):
    """Initialize Pinecone connection and index."""
    pc = init_pinecone()
    index = create_or_load_index(pc, index_name)

    # Check if index is populated
    stats = index.describe_index_stats()
    vector_count = stats.get('total_vector_count', 0)

    return index, vector_count


def setup_embeddings():
    """Load embeddings model."""
    return download_embeddings()


def populate_vector_store(embedding, data_directory="data", index_name="medical-chatbot-1"):
    """Populate vector store with PDF documents if empty."""
    # Check if data directory exists
    if not os.path.exists(data_directory):
        return False, "Data directory not found! Please create a 'data' folder with PDF files."

    # Load and process PDFs
    documents = load_pdf_files(data_directory)

    if not documents:
        return False, "No PDF files found in the data directory!"

    # Split into chunks
    text_chunks = text_split(documents)

    # Create vector store
    create_vectorstore(text_chunks, embedding, index_name)

    return True, f"Successfully processed {len(documents)} documents into {len(text_chunks)} chunks"


def create_vector_store_connection(embedding, index_name="medical-chatbot-1"):
    """Create vector store connection."""
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embedding
    )


def create_retriever(docsearch, score_threshold=0.75):
    """Create retriever with similarity search."""
    return docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": score_threshold}
    )


def create_rag_chain(retriever):
    """Create the complete RAG chain."""
    # Initialize LLM
    chat_model = get_chat_model()

    # Create RAG chain using prompt
    prompt = qa_prompt
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def initialize_rag_system(index_name="medical-chatbot-1", data_directory="data"):
    """Initialize the complete RAG system."""
    try:
        # Validate API keys
        api_valid, api_error = validate_api_keys()
        if not api_valid:
            return None, f"API validation failed: {api_error}"

        # Setup Pinecone
        index, vector_count = setup_pinecone_index(index_name)

        # Load embeddings
        embedding = setup_embeddings()

        # Handle empty index
        if vector_count == 0:
            success, message = populate_vector_store(embedding, data_directory, index_name)
            if not success:
                return None, f"Failed to populate vector store: {message}"

        # Create vector store connection
        docsearch = create_vector_store_connection(embedding, index_name)

        # Create retriever
        retriever = create_retriever(docsearch)

        # Create RAG chain
        rag_chain = create_rag_chain(retriever)

        return rag_chain, f"RAG system initialized successfully. Vectors in index: {vector_count}"

    except Exception as e:
        return None, f"Error initializing RAG system: {str(e)}"


def get_rag_response(rag_chain, query):
    """Get response from RAG chain with error handling."""
    try:
        response = rag_chain.invoke({"input": query})
        answer = response.get("answer", "")

        # Check if any relevant documents were retrieved
        if "context" in response and response["context"]:
            return {
                "success": True,
                "answer": answer,
                "context": response["context"]
            }
        else:
            # No relevant documents found
            no_docs_msg = (
                "I couldn't find relevant information in the medical knowledge base "
                "for your query. Please try rephrasing your question or consult a "
                "healthcare professional for specific medical advice."
            )
            return {
                "success": False,
                "error": no_docs_msg,
                "no_context": True
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "no_context": False
        }
