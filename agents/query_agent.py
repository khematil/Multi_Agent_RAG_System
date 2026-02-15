from agents.state import AgentState
from utils import get_qdrant_client, get_embedding_model
from config import COLLECTION_NAME

def query_agent(state: AgentState) -> AgentState:
    print("\n🔍 ========== QUERY AGENT ENTRY ==========")
    query = state['query']
    
    state["messages"].append(f"[Query Agent]: Processing '{query}'")
    
    client = get_qdrant_client()
    model = get_embedding_model()
    
    query_vector = model.encode([query])[0]
    
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=5
    )
    
    retrieved_chunks = []
    for result in results.points:
        retrieved_chunks.append({
            'text': result.payload.get('text', ''),
            'source': result.payload.get('source_file'),
            'score': result.score,
            'chunk_index': result.payload.get('chunk_index', 0)
        })
        
    state['retrieved_chunks'] = retrieved_chunks
    state['current_step'] = 'response'
    state["messages"].append(f"[Query Agent]: Found {len(retrieved_chunks)} chunks.")
    
    return state