from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
#from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from config import (
    QDRANT_PATH, 
    EMBEDDING_MODEL,
    # ANTHROPIC_API_KEY,
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMP,
    LLM_MAX_TOKENS,
    HF_TOKEN
)

def get_qdrant_client():
    """Get Qdrant client."""
    return QdrantClient(path=QDRANT_PATH)

def get_embedding_model():
    """Get embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL, token=HF_TOKEN)

def get_llm():
    """Get LLM"""
    return ChatGroq(
        model = LLM_MODEL,
        temperature = LLM_TEMP,
        max_tokens = LLM_MAX_TOKENS,
        api_key = GROQ_API_KEY
    )
    