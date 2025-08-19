# helper functions for data processing, etc.
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

def load_pdf_files(data_path: str):
    """Load PDF documents from a given directory."""
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def text_split(documents, chunk_size=500, chunk_overlap=20):
    """Split documents into smaller text chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def download_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load HuggingFace embeddings model (default: MiniLM-L6-v2)."""
    return HuggingFaceEmbeddings(model_name=model_name)

def init_pinecone():
    """Initialize Pinecone client using API key from environment."""
    return Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def create_or_load_index(pc, index_name="medical-chatbot-1"):
    """Create Pinecone index if it doesn’t exist, then return it."""
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

def create_vectorstore(chunks, embedding, index_name="medical-chatbot-1"):
    """Embed and push chunks into Pinecone, return the vectorstore."""
    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        index_name=index_name
    )

