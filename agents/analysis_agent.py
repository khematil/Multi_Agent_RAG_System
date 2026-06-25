from typing import Dict, List
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
        state['generate_response'] = False
        state["analysis_reason"] = "No relevant documents found in database"
        state["final_answer"] = generate_response(query, chunks, 0.0)
        state["messages"].append("[Analysis Agent]: No chunks retrieved - low confidence")
        print("No results found - skipping [Response Agent]")
        return state
    
     # Case 2: Check similarity scores
    top_score = chunks[0]["score"]  
    avg_score = sum(c["score"] for c in chunks[:3]) / min(3, len(chunks))
    
    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        # Good quality --> proceed to response
        state["retrieval_quality"] = "good"
        state["confidence_score"] = top_score
        state["generate_response"] = True
        state["analysis_reason"] = f"High similarity score ({top_score:.3f}), relevant context found"
        state["messages"].append(f"[Analysis Agent]: High confidence ({top_score:.3f}) - proceeding to response")
        
    elif top_score >= LOW_CONFIDENCE_THRESHOLD:
        # Medium quality --> still answer but with lower confidence
        state["retrieval_quality"] = "medium"
        state["confidence_score"] = avg_score
        state["generate_response"] = True
        state["analysis_reason"] = f"Medium similarity scores (avg: {avg_score:.3f}), partial context"
        state["messages"].append(f"[Analysis Agent]: Medium confidence ({avg_score:.3f}) - will answer with caveats")
        
    else:
        # Poor quality --> tell user we don't have good info
        state["retrieval_quality"] = "poor"
        state["confidence_score"] = top_score
        state["generate_response"] = False
        state["analysis_reason"] = f"Low similarity scores (top: {top_score:.3f}), weak relevance"
    
        state["final_answer"] = generate_response(query, chunks, top_score)
        state["messages"].append(f"[Analysis Agent]: Low confidence ({top_score:.3f}) - insufficient context")   
        print("Poor quality - skipping [Response Agent]")
   
    print(f"Quality: {state['retrieval_quality']} | Confidence: {state['confidence_score']:.3f}")    
    return state


def generate_response(query: str, chunks: List[Dict], score: float):
    
    if score == 0.0:
        response = f"""Couldn't find relevant documents in my knowledge base to answer your question. 
        
        **Your question:** "{query}"
        
        **Possible reasons:**
        - This topic might not be covered in the available documents
        - The question might use different terminology than what's in the database
        - The relevant documents may not have been added yet
        
        **Suggestions:**
        - Try rephrasing your question using different keywords
        - Check if the relevant documents have been ingested into the system
        """
        return response
    
    else: 
        sources = ", ".join(set(c['source'] for c in chunks[:3]))
        
        response = f"""Found some documents, but they don't seem to be highly relevant to your question.
 
         **Your question:** "{query}"       

        **What I found:**
        - Retrieved {len(chunks)} document chunks
        - Highest relevance score: {score:.3f} (below confidence threshold of 0.4)
        - Sources checked: {sources}
        
        **Analysis:**
        The retrieved documents have weak relevance to your question. The content doesn't closely match what you're asking about.

        **Suggestions:**
        - Rephrase your question to be more specific
        - Try asking about topics that are covered in the documents
        - Check if documents related to your question have been added

        *If you believe this topic should be in the database, you may want to check if the documents were successfully ingested.*        
        """
        
        