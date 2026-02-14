from agents.state import AgentState
from agents.query_agent import query_agent
from agents.response_agent import response_agent

def test_query_agent():
    
    initial_state: AgentState = {
        "query": "What is the CAP theorem?",
        "retrieved_chunks": [],
        "messages": [],
        "current_step": "query"
    }
    
    print("--- Testing Query Agent with State ---")
    print("\nInitial State:")
    print(f"\tQuery: {initial_state['query']}")
    print(f"\tStep: {initial_state['current_step']}")
    print(f"\tChunks: {len(initial_state['retrieved_chunks'])}")

    print("\n--- Running Query Agent ---")
    
    result_state = query_agent(initial_state)
    
    print("\n--- Result State ---")
    print(f"\n\tStep: {result_state['current_step']}")
    print(f"\tChunks retrieved: {len(result_state['retrieved_chunks'])}")
    
    
    print("\n--- Top 3 Chunks ---")
    for i, chunk in enumerate(result_state["retrieved_chunks"][:3], 1):
        print(f"\n\t [{i} - Score: {chunk['score']:.3f}")
        print(f"\tSource: {chunk['source']}")
        print(f"\tText: {chunk['text'][:100]}. . .")
        
    print("\n--- Agent Messages ---")
    for msg in result_state['messages']:
        print(f"\t{msg}")
    
    print("\n--- Test Complete ---")
    
    
def test_two_agents():
    
    initial_state: AgentState = {
        "query": "What is the CAP theorem?",
        "retrieved_chunks": [],
        "messages": [],
        "current_step": "query"
    }
    
    print("=== Testing 2 Agent Flow: [QUERY ---> RESPONSE] ===")
    print("\nInitial State:")
    print(f"\tQuery: {initial_state['query']}")
    print(f"\tStep: {initial_state['current_step']}")
    print(f"\tChunks: {len(initial_state['retrieved_chunks'])}")

    print("\n=== STEP 1: Running Query Agent (RAG Retrieval) ===")
    state = query_agent(initial_state)
    
 
    print(f"\tChunks retrieved: {len(state['retrieved_chunks'])}")
    print(f"\n\tNext Step: {state['current_step']}")
  
    print("\n=== Query Agent Messages ===")
    for msg in state['messages']:
        print(f"\t{msg}")
    
    print("\n=== STEP 2: Response Agent (Claude AI) ===")
    state = response_agent(state)
    print(f"Answer generated . . .")
    print(f"Current Step: {state['current_step']}")

    print("\n=== Response Agent Answer ===")
    print(f"\nQuestion: {state['query']}")
    print(f"\nAnswer: {state['final_answer']}")
    print(f"\nSources Used:")
    for i, chunk in enumerate(state['retrieved_chunks'][:3], 1):
        print(f"\t[{i}] {chunk['source']} (chunk {chunk['chunk_index']}) - Score: {chunk['score']:.3f}")
       
       
    print("\n=== Test Complete ===")
    
if __name__ == "__main__":
    ##test_query_agent()   
    test_two_agents()