import os
import platform
import logging
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.absolute()

if Path("/tmp").exists() and os.name == "posix":
    DATA_DIR = "/tmp"
else:
    DATA_DIR = str(PROJECT_ROOT / "data")
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


BASE_VERCEL_URL = os.getenv("BASE_VERCEL_URL")
VERCEL_BYPASS = os.getenv("VERCEL_BYPASS_SECRET")

QDRANT_URL = os.getenv("QDRANT_ENDPOINT") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Collection configuration
COLLECTION_NAME = "documents"

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

## LLM configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "openai/gpt-oss-20b"
LLM_TEMP = 0.0
LLM_MAX_TOKENS = 1024

## Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")

# Chunk score: Cosine similarity thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.7  
LOW_CONFIDENCE_THRESHOLD = 0.4  
