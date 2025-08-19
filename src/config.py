import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()

def get_env_var(key: str) -> str:
    """Helper to fetch environment variables safely."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"⚠️ Missing environment variable: {key}")
    return value

def get_chat_model():
    """Initialize and return the chat model (DeepSeek via OpenRouter)."""
    return ChatOpenAI(
        model="deepseek/deepseek-chat",              # You can swap this easily later
        api_key=SecretStr(get_env_var("OPENROUTER_API_KEY")),   # Pulled from .env 
        base_url="https://openrouter.ai/api/v1",     # OpenRouter endpoint
        timeout=30,                                  # Avoids infinite hangs
        max_retries=1,                               # Fail fast instead of looping
        default_headers={
            "HTTP-Referer": "http://localhost:8501", # For OpenRouter analytics
            "X-Title": "Medical Chatbot 1"
        },
    )
