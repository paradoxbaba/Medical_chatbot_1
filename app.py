import streamlit as st
import os
from src.helper import (
    load_pdf_files, 
    text_split, 
    download_embeddings, 
    init_pinecone, 
    create_or_load_index, 
    create_vectorstore
)
from src.config import get_env_var, get_chat_model
from src.prompt import system_prompt, qa_prompt  # assuming these exist
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page configuration
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Chatbot")
st.markdown("Ask me any medical questions and I'll help you with first-aid information!")

@st.cache_resource
def initialize_system():
    """Initialize the RAG system with caching to avoid reloading."""
    
    # Check API keys using config helper
    try:
        PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
        # OPENROUTER_API_KEY will be handled by get_chat_model()
    except ValueError as e:
        st.error(str(e))
        st.stop()
    
    # Initialize Pinecone
    with st.spinner("Connecting to Pinecone..."):
        pc = init_pinecone()
        index = create_or_load_index(pc, "medical-chatbot-1")
    
    # Check if index is populated
    stats = index.describe_index_stats()
    vector_count = stats.get('total_vector_count', 0)
    
    st.write(f"📊 Current vectors in index: {vector_count}")
    
    # Load embeddings
    with st.spinner("Loading embeddings model..."):
        embedding = download_embeddings()
    
    # If index is empty, populate it
    if vector_count == 0:
        st.warning("Index is empty. Loading and processing PDF files...")
        
        # Check if data directory exists
        if not os.path.exists("data"):
            st.error("Data directory not found! Please create a 'data' folder with PDF files.")
            st.stop()
        
        # Load and process PDFs
        with st.spinner("Loading PDF files..."):
            documents = load_pdf_files("data")
        
        if not documents:
            st.error("No PDF files found in the data directory!")
            st.stop()
        
        st.write(f"📄 Loaded {len(documents)} document pages")
        
        # Split into chunks
        with st.spinner("Splitting documents into chunks..."):
            text_chunks = text_split(documents)
        
        st.write(f"📝 Created {len(text_chunks)} text chunks")
        
        # Create vector store (only when empty)
        with st.spinner("Creating vector embeddings and storing in Pinecone..."):
            docsearch = create_vectorstore(text_chunks, embedding, "medical-chatbot-1")
        
        st.success("✅ Vector store created successfully!")
    
    else:
        # Use existing vector store (don't recreate)
        st.success("✅ Using existing vector store")
    
    # Always create the vector store connection (whether new or existing)
    docsearch = PineconeVectorStore(
        index_name="medical-chatbot-1",
        embedding=embedding
    )
    
    # Create retriever
    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": 0.75}
    )
    
    # Initialize LLM using config
    with st.spinner("Initializing chat model..."):
        chat_model = get_chat_model()
    
    # Create RAG chain using prompt from src.prompt
    prompt = qa_prompt
    
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Initialize the system
try:
    rag_chain = initialize_system()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about medical conditions, symptoms, or first-aid..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching medical knowledge base..."):
                try:
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    # Check if any relevant documents were retrieved
                    if "context" in response and response["context"]:
                        st.markdown(answer)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Show source documents in an expander (optional)
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(response["context"]):
                                st.write(f"**Source {i+1}:**")
                                st.write(doc.page_content[:200] + "...")
                                st.write(f"*Page: {doc.metadata.get('page', 'Unknown')}*")
                                st.divider()
                    else:
                        # No relevant documents found
                        no_docs_msg = ("I couldn't find relevant information in the medical knowledge base "
                                     "for your query. Please try rephrasing your question or consult a "
                                     "healthcare professional for specific medical advice.")
                        st.warning(no_docs_msg)
                        st.session_state.messages.append({"role": "assistant", "content": no_docs_msg})
                
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

except Exception as e:
    st.error(f"Failed to initialize the medical chatbot: {str(e)}")
    st.info("Please check your API keys and data directory.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This medical chatbot uses:
    - **Vector Database**: Pinecone
    - **Embeddings**: HuggingFace MiniLM-L6-v2
    - **LLM**: DeepSeek Chat via OpenRouter
    - **Framework**: LangChain RAG
    
    **⚠️ Disclaimer**: This is for informational purposes only. 
    Always consult healthcare professionals for medical advice.
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()