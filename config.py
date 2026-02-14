import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_DIR = "./data"
QDRANT_PATH = "./qdrant_db"

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Collection configuration
COLLECTION_NAME = "documents"

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# LLM configuration
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# LLM_MODEL = "claude-haiku-4-5-20251001"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "openai/gpt-oss-20b"
LLM_TEMP = 0.0
LLM_MAX_TOKENS = 1024