import streamlit as st
from src.rag import initialize_rag_system, get_rag_response

# Page configuration
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Chatbot")
st.markdown("Ask me any medical questions and I'll help you with first-aid information!")

@st.cache_resource
def get_rag_chain():
    """Initialize the RAG system with caching."""
    with st.spinner("Initializing RAG system..."):
        rag_chain, message = initialize_rag_system()

        if rag_chain:
            st.success(f"✅ {message}")
            return rag_chain
        else:
            st.error(f"❌ {message}")
            return None

def main():
    try:
        # Initialize RAG system
        rag_chain = get_rag_chain()

        if not rag_chain:
            st.error("Failed to initialize RAG system")
            st.stop()

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
                    response = get_rag_response(rag_chain, prompt)

                    if response["success"]:
                        st.markdown(response["answer"])
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response["answer"]
                        })

                        # Show source documents
                        if response.get("context"):
                            with st.expander("📚 Source Documents"):
                                for i, doc in enumerate(response["context"]):
                                    st.write(f"**Source {i+1}:**")
                                    st.write(doc.page_content[:200] + "...")
                                    st.write(f"*Page: {doc.metadata.get('page', 'Unknown')}*")
                                    st.divider()
                    else:
                        error_msg = response.get("error", "An unknown error occurred")
                        if response.get("no_context"):
                            st.warning(error_msg)
                        else:
                            st.error(f"Sorry, I encountered an error: {error_msg}")

                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })

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
                
    **📝 Source Code**: [GitHub](https://github.com/paradoxbaba/Medical_chatbot_1)
    """)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
