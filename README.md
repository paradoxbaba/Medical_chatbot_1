# 🏥 First Aid Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain and Streamlit, designed to provide immediate, context-specific first-aid guidance by querying a knowledge base of medical PDFs. It uses Pinecone as a vector database and a powerful LLM through OpenRouter to deliver accurate, sourced answers.

**⚠️ Important Disclaimer**: This application is for informational and educational purposes only. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always seek the advice of qualified healthcare providers with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you read in this chatbot.

## ✨ Features

- **Document Ingestion**: Automatically loads and processes PDF documents from a `data/` directory.
- **Intelligent Chunking**: Splits documents into meaningful chunks for better retrieval accuracy.
- **Vector Search**: Utilizes Pinecone to create and query a vector store of medical information efficiently.
- **Safety-First Prompting**: The system prompt is carefully engineered to prioritize safety, instructing the AI to *only* use the provided context and to explicitly state when information is unavailable.
- **Clean Web Interface**: A user-friendly Streamlit app with a chat interface and a sidebar showing source documents for transparency.
- **Error Handling**: Robust error handling for missing API keys, empty vector stores, and failed queries.

## 🛠️ Tech Stack

- **Framework**: `LangChain`
- **Vector Database**: `Pinecone`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` from HuggingFace
- **LLM**: `DeepSeek Chat` (via OpenRouter)
- **Web Interface**: `Streamlit`
- **Language**: `Python 3.10+`

## 📋 Prerequisites

Before you begin, ensure you have the following:
1.  **Python 3.10+** installed on your system.
2.  A **Pinecone account** ([https://www.pinecone.io/](https://www.pinecone.io/)). Get your API key from the console.
3.  An **OpenRouter account** ([https://openrouter.ai/](https://openrouter.ai/)). Get your API key and ensure you have credits to use the DeepSeek model.

## 🚀 Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/paradoxbaba/Medical_chatbot_1
    cd Medical_chatbot_1
    ```

2.  **Create a virtual environment and activate it**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**
    Create a `.env` file in the root directory of the project and add your API keys:
    ```ini
    PINECONE_API_KEY=your_pinecone_api_key_here
    OPENROUTER_API_KEY=your_openrouter_api_key_here
    ```

5.  **Add your PDF documents**
    Create a `data/` folder in the root directory and place your first-aid or medical PDF files inside it.
    ```bash
    mkdir data
    # Copy your PDFs into the ./data folder
    ```

6.  **Run the application**
    ```bash
    streamlit run app.py
    ```
    The application will automatically open in your default web browser.

## 📁 Project Structure

```
Medical_chatbot_1/
├── src/
│   ├── __init__.py
│   ├── helper.py          # Functions for loading, splitting, embedding PDFs
│   ├── prompt.py          # System prompt and LangChain prompt template
│   ├── rag.py             # Core RAG chain setup and query logic
│   └── config.py          # (Inferred) Configuration & API key management
├── data/                  # Directory for PDF files (you create this)
├── .env                   # File for environment variables (you create this)
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md
```

## 🧠 How It Works

1.  **Initialization**: When you first run the app, it checks the Pinecone index. If the index is empty, it processes all PDFs in the `data/` folder.
2.  **Processing**: Each PDF is loaded, split into chunks, converted into vector embeddings (using HuggingFace's MiniLM model), and stored in Pinecone.
3.  **Querying**: When you ask a question, the system converts your query into an embedding and performs a similarity search in the Pinecone vector store to find the most relevant text chunks.
4.  **Answering**: These relevant chunks are passed, along with your question and a strict system prompt, to the DeepSeek LLM via OpenRouter to generate a context-aware, safe response.
5.  **Display**: The answer is displayed in the chat interface, and the source text chunks are available for viewing to ensure transparency.

## 🔧 Configuration

Key configuration points can be found in the code:
- **Pinecone Index Name**: Default is `medical-chatbot-1`. Change it in `rag.py`'s function definitions.
- **Embedding Model**: Default is `sentence-transformers/all-MiniLM-L6-v2`. Change it in `helper.py` in the `download_embeddings()` function.
- **Similarity Score Threshold**: The minimum confidence score for a chunk to be considered relevant. Adjust the `score_threshold` parameter in the `create_retriever()` function in `rag.py`.

## 👨‍💻 Author

**Animesh (paradoxbaba)**
- GitHub: [https://github.com/paradoxbaba](https://github.com/paradoxbaba)

## 📄 License

This project is licensed for educational use. Please ensure you comply with the licenses of any medical documents you use to populate the knowledge base.