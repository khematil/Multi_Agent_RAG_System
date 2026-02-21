from fastapi import FastAPI

from contextlib import asynccontextmanager
import logging
from app.routers.query import router as query_router
from utils import get_qdrant_client, get_embedding_model, get_llm
from config import DATA_DIR, QDRANT_PATH, COLLECTION_NAME


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API...")
    try:
        logger.info("Loading Qdrant client...")
        app.state.qdrant_client = get_qdrant_client()
        logger.info(f"Qdrant client loaded (path: {QDRANT_PATH})")
        
        logger.info("Loading embedding model...")
        app.state.embedding_model = get_embedding_model()
        logger.info("Embedding model loaded")
        
        logger.info("Loading LLM client...")
        app.state.llm = get_llm()
        logger.info("LLM client loaded")
        
        # Store configuration
        app.state.config = {
            "data_dir": DATA_DIR,
            "collection_name": COLLECTION_NAME
        }
        
        logger.info("API ready!")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    
    yield
    
    logger.info("Shutting downAPI...")
    
    
    logger.info("Shutdown complete")

app = FastAPI()

app.include_router(
    query_router,
    prefix="/api"
)
@app.get("/")
async def root():
    return {
        "message": "Welcome to Kyle's RAG Multi-Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/query/health"
    }


@app.get("/health")
async def health_check():
    """
    Global health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "RAG Multi Agent System API"
    }

# @app.post("/api/query")
# async def query_rag(query: str):
#     return 

# @app.get("/health")
# async def get_server_status():
#     return

# @app.post("/api/ingest")
# async def ingest_documents(file):
#     return