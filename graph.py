from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from agents.state import AgentState
from agents.query_agent import query_agent
from agents.response_agent import response_agent
from agents.analysis_agent import analysis_agent


def create_graph():
    
    # Init StateGraph
    builder = StateGraph(AgentState)
    
    # Nodes
    builder.add_node("query", query_agent)
    builder.add_node("analysis", analysis_agent)
    builder.add_node("response", response_agent)
    
    # Workflow
    builder.set_entry_point("query")            # START at QUERY
    builder.add_edge("query", "analysis")       # QUERY -> ANALYSIS
    
    builder.add_conditional_edges(              # ANALYSIS -> RESPONSE or ANALYSIS -> END
        "analysis",
        should_generate_response,
        {
            "response" : "response",
            END : END   
        }
    )
    
    builder.add_edge("response", END)           # RESPOSNE -> END
    
    app = builder.compile()
    
    return app

def run_rag_query(query: str):
    app = create_graph()
    
    initial_state: AgentState = {
        "query": query,
        "retrieved_chunks": [],
        "retrieval_quality": "",
        "confidence_score": 0.0,
        "analysis_reason": "",
        "final_answer": "",
        "messages": [],
        "current_step": "query"
    }
    
    final_state = app.invoke(initial_state)
    
    return final_state

def should_generate_response(state: AgentState) -> str:
    if state.get("generate_response", False):
        print("[Router]: Quality sufficient -> Response Agent")
        return "response"           
    else:
        print("[Router]: Quality insufficient -> Skip to END")
        return END
    
def display_graph():
    app = create_graph()
    
    graph_visual = app.get_graph().draw_mermaid_png()
    
    with open("graph_visual.png", "wb") as f:
        f.write(graph_visual)
        
if __name__ == "__main__":
    display_graph()