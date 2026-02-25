from fastapi import APIRouter, HTTPException
from api.models import QueryRequest, QueryResponse
from graph import run_rag_query
import time

router = APIRouter(prefix="/query", tags=["query"])

@router.post("/", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    
    start_time = time.time()
    
    try:
        final_state = run_rag_query(request.question)
        
        sources = []
        if request.include_sources:
            sources = [
                {
                    "source": c["source"],
                    "chunk_index": c["chunk_index"],
                    "score": c["score"]
                }
                for c in final_state["retrieved_chunks"][:request.max_results]
            ]
            
        response = QueryResponse(
            answer=final_state["final_answer"],
            confidence_score=final_state['confidence_score'],
            retrieval_quality=final_state["retrieval_quality"],
            sources = sources if request.include_sources else None,
            processing_time= time.time() - start_time
            
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

