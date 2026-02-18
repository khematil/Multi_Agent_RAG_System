from typing import TypedDict, List, Annotated, Dict

class AgentState(TypedDict):
    
    # User Input
    query: str
    
    # RAG retrieval (Query Agent)
    retrieved_chunks: List[Dict[str, any]]
    
    # Analysis agent
    generate_response: bool
    retrieval_quality: str
    confidence_score: float
    analysis_reason: str
    
    # Response Agent
    final_answer: str
    
    
    # Agent coordination
    messages: List[str]
    current_step: str