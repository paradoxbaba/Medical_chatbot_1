import streamlit as st
from src.helper import init_pinecone, create_or_load_index
from src.config import get_chat_model
from src.prompt import qa_prompt
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")

st.title("🩺 Medical Chatbot")
st.write("Welcome! Ask me medical-related questions (educational purposes only).")

# --- CACHED SETUP ---
@st.cache_resource
def setup_rag_pipeline():
    # Connect to Pinecone index (already built earlier in notebook)
    pc = init_pinecone()
    index = create_or_load_index(pc, index_name="medical-chatbot-1")

    # Load embeddings (must match what you used earlier in notebook!)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load existing vectorstore
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot-1",
        embedding=embeddings
    )

    # Build retriever + chain
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.75},
    )

    chat_model = get_chat_model()
    doc_chain = create_stuff_documents_chain(chat_model, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    return rag_chain

with st.spinner("🔄 Connecting to Pinecone & loading model..."):
    rag_chain = setup_rag_pipeline()

# --- UI Input/Output ---
user_input = st.text_input("Enter your medical question:")

if user_input:
    with st.spinner("🤔 Thinking..."):
        response = rag_chain.invoke({"input": user_input})
    st.write(f"🤖 {response['answer']}")
