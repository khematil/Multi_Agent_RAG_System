from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from agents.state import AgentState
from agents.query_agent import query_agent
from agents.response_agent import response_agent


def create_graph():
    
    # Init StateGraph
    builder = StateGraph(AgentState)
    
    # Nodes
    builder.add_node("query", query_agent)
    builder.add_node("response", response_agent)
    
    # Workflow
    builder.set_entry_point("query")        # START at QUERY
    builder.add_edge("query", "response")   # QUERY -> RESPONSE
    builder.add_edge("response", END)       # RESPOSNE -> END
    
    app = builder.compile()
    
    return app

def run_rag_query(query: str):
    app = create_graph()
    
    initial_state: AgentState = {
        "query": query,
        "retrieved_chunks": [],
        "final_answer": "",
        "messages": [],
        "current_step": "query"
    }
    
    final_state = app.invoke(initial_state)
    
    return final_state
    


def display_graph():
    app = create_graph()
    
    graph_visual = app.get_graph().draw_mermaid_png()
    
    with open("graph_visual.png", "wb") as f:
        f.write(graph_visual)


if __name__ == "__main__":
    
    query = "What is the CAP theorem?"
    
    print(f"\nQuestion: {query}\n")
    
    result = run_rag_query(query)
    
    print(f"\nAnswer:")
    print(f"{result['final_answer']}\n")
    
    print(f"Sources:")
    for i, chunk in enumerate(result['retrieved_chunks'][:3], 1):
        print(f"  [{i}] {chunk['source']} (chunk {chunk['chunk_index']}) - {chunk['score']:.3f}")
    
    print(f"\nAgent Flow:")
    for msg in result['messages']:
        print(f"{msg}") 
    


   # display_graph()