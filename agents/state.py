from typing import TypedDict, List, Annotated, Dict
from operator import add

class AgentState(TypedDict):
    query: str
    
    retrieved_chunks: List[Dict[str, any]]
    
    final_answer: str
    
    messages: List[str]
    
    current_step: str