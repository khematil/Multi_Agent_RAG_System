from typing import TypedDict, List, Annotated, Dict
from operator import add

class AgentState(TypedDict):
    query: str
    
    retrieved_chunks: List[Dict[str, any]]
    
    final_answer: str
    
    messages: Annotated[List[str], add]
    
    current_step: str