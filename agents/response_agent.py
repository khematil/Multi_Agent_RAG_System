from agents.state import AgentState
from utils import get_llm

def response_agent(state: AgentState) -> AgentState:
    query = state['query']
    chunks = state['retrieved_chunks']
    quality = state.get('retrieval_quality', 'unknown')
    confidence = state.get('confidence_score', 0.0)
    
    state['messages'].append("[Response Agent]: Generating response . . .")

    # Case 1: No results / poor score
    if not chunks:
        state['messages'].append("[Response Agent]: No chunks available")
        state['current_step'] = 'complete'
        return state
    
    if quality in ['no results', 'poor']:

        final_answer = f"""I don't have enough relevant information in my knowledge base to answer this question confidently.

        The question was: "{query}"

        Reason: {state.get('analysis_reason', 'No matching documents found')}

        You might try:
        - Rephrasing your question
        - Asking about a different topic that might be in the available documents
        - Checking if the relevant documents have been added to the system
        
        """
        state["final_answer"] = final_answer
        state['messages'].append("[Response Agent]: Generated low-confidence response")
        return state
    
    
    
    # Case 2: Good / moderate quality --> Generate an answer
    context_parts = []
    
    for i, chunk in enumerate(chunks[:3], 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Chunk{chunk['chunk_index']}]\n"
            f"{chunk['text']}\n"
        )
        
    context = "\n".join(context_parts)

    confidence_instruction = ""
    
    if quality == "medium":
        confidence_instruction = """\nNote the retrieved context has medium relevance. 
                                    If the context doesn\'t fully answer the question, 
                                    acknowledge what\'s missing.
                                """
                                
    prompt = f"""You are a helpful AI assistant answering questions based on the provided context.
    
    Context from retrieved documents: {context}
    
    User Question: {query}
    
    Instructions:
    
    1. Answer the questions based ONLY on the information in the context provided.
    2. Be concise and accurate.
    3. If the context does not contain enough information to fully answer the questions, say that there is not enough information.
    4. Cite which source(s) you used in your answer.{confidence_instruction}
    
    Answer:
    """
        
    llm = get_llm()
    
    try:
        response = llm.invoke(prompt)
        final_answer = response.content
        
        if quality == "medium":
            final_answer += f"\n\n*Note: This answer is based on partially relevant context (confidence: {confidence:.2f}). Some details may be incomplete.*"
            
        state['messages'].append("[Response Agent]: Answer generated successfully.")
        
    except Exception as e:
        final_answer = f"Error generating a response: {e}"
        state['messages'].append(f"[Response Agent]: Error - {e}.")
    
    state["final_answer"] = final_answer
    state["current_step"] = "complete"
    
    
    return state
        

    
    