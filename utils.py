from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
#from langchain_anthropic import ChatAnthropic

import logging

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

_qdrant_client = None
_embedding_model = None
_llm = None

def get_qdrant_client():
    global _qdrant_client
    
    if _qdrant_client is None:
        print("Loading Qdrant client...")
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    
    return _qdrant_client

def get_embedding_model():
    global _embedding_model
    
    if _embedding_model is None:
        print("Loading embedding model...")
        logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
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
    