from qdrant_client import QdrantClient
import logging
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMP,
    LLM_MAX_TOKENS,
    HF_TOKEN
)

_qdrant_client = None
_embedding_model = None
_llm = None

def get_qdrant_client():
    global _qdrant_client
    
    if _qdrant_client is None:
        print("Loading Qdrant client...")
        _qdrant_client = QdrantClient(url=QDRANT_URL,
                                      api_key=QDRANT_API_KEY
                                    )
        print("Qdrant cloud connected ... ")
    
    return _qdrant_client

def get_embedding_model():
    global _embedding_model
    
    if _embedding_model is None:
        print("Loading embedding model...")
        print(f"Connecting to Hugging Face Inference API for {EMBEDDING_MODEL}...")
    
        _embedding_model = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=HF_TOKEN,
            model=EMBEDDING_MODEL
        )
    
    return _embedding_model

def get_llm():
    global _llm
    
    if _llm is None:
        print("Loading LLM...")
        _llm = ChatGroq(
            model = LLM_MODEL,
            temperature = LLM_TEMP,
            max_tokens = LLM_MAX_TOKENS,
            api_key = GROQ_API_KEY
        )
    
    return _llm
    