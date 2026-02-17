from agents.state import AgentState
from config import HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD

def analysis_agent(state: AgentState) -> AgentState:
    
    chunks = state['retrieved_chunks']
    query = state['query']
    
    
    
    state["messages"].append(f"[Analysis Agent]: Evaluating {len(chunks)} retrieved chunks")
    
    # Case 1: No results
    if not chunks:
        state["retrieval_quality"] = "no_results"
        state["confidence_score"] = 0.0
        state["analysis_reason"] = "No relevant documents found in database"
        state["messages"].append("[Analysis Agent]: No chunks retrieved - low confidence")
        return state
    
     # Case 2: Check similarity scores
    top_score = chunks[0]["score"]  
    avg_score = sum(c["score"] for c in chunks[:3]) / min(3, len(chunks))
    

    
    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        # Good quality --> proceed to response
        state["retrieval_quality"] = "good"
        state["confidence_score"] = top_score
        state["analysis_reason"] = f"High similarity score ({top_score:.3f}), relevant context found"
        state["messages"].append(f"[Analysis Agent]: High confidence ({top_score:.3f}) - proceeding to response")
        
    elif top_score >= LOW_CONFIDENCE_THRESHOLD:
        # Medium quality --> still answer but with lower confidence
        state["retrieval_quality"] = "medium"
        state["confidence_score"] = avg_score
        state["analysis_reason"] = f"Medium similarity scores (avg: {avg_score:.3f}), partial context"
        state["messages"].append(f"[Analysis Agent]: Medium confidence ({avg_score:.3f}) - will answer with caveats")
        
    else:
        # Poor quality --> tell user we don't have good info
        state["retrieval_quality"] = "poor"
        state["confidence_score"] = top_score
        state["analysis_reason"] = f"Low similarity scores (top: {top_score:.3f}), weak relevance"
        state["messages"].append(f"[Analysis Agent]: Low confidence ({top_score:.3f}) - insufficient context")   
    
    print(f"Quality: {state['retrieval_quality']} | Confidence: {state['confidence_score']:.3f}")    
    return state