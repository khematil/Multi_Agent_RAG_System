from agents.state import AgentState
from utils import get_llm

def response_agent(state: AgentState) -> AgentState:
    print("\n🤖 ========== RESPONSE AGENT ENTRY ==========")   
    query = state['query']
    chunks = state['retrieved_chunks']
    
    state['messages'].append("[Response Agent]: Generating response with Claude AI")
    
    if not chunks:
        state['messages'].append("[Response Agent]: No chunks available")
        state['current_step'] = 'complete'
        return state
    
    context_parts = []
    
    for i, chunk in enumerate(chunks[:3], 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Chunk{chunk['chunk_index']}]\n"
            f"{chunk['text']}\n"
        )
        
    context = "\n".join(context_parts)
    
    prompt = f"""You are a helpful AI assistant answering questions based on the provided context.
    
    Context from retrieved documents: {context}
    
    User Question: {query}
    
    Instructions:
    
    1. Answer the questions based ONLY on the information in the context provided.
    2. Be concise and accurate.
    3. If the context does not contain enough information to fully answer the questions, say that there is not enough information.
    4. Cite which source(s) you used in your answer.
    
    Answer:
    """
    
    llm = get_llm()
    
    try:
        response = llm.invoke(prompt)
        final_answer = response.content
        state['messages'].append("[Response Agent]: Answer generated successfully.")
        
    except Exception as e:
        final_answer = f"Error generating a response: {e}"
        state['messages'].append(f"[Response Agent]: Error - {e}.")
    
    state["final_answer"] = final_answer
    state["current_step"] = "complete"
    
    
    return state
        

    
    
