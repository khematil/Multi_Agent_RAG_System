from langgraph.graph import StateGraph, END
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
    builder.set_entry_point("query")
    builder.add_edge("query", "response")
    builder.add_edge("response", END)
    
    app = builder.compile()
    
    return app
