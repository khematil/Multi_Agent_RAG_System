import pytest
from agents.state import AgentState
from agents.query_agent import query_agent
from agents.analysis_agent import analysis_agent
from agents.response_agent import response_agent
from graph import run_rag_query

class TestQueryAgent:
    def test_query_agent_retrieves_chunks(self):
        
        # ARRANGE
        initial_state: AgentState = {
            "query": "What is the CAP theorem?",
            "retrieved_chunks": [],
            "retrieval_quality": "",
            "confidence_score": 0.0,
            "analysis_reason": "",
            "final_answer": "",
            "should_generate_response": False,
            "messages": [],
            "current_step": "query"
        }
        
        # ACT
        result_state = query_agent(initial_state)
        
        # ASSERT
        assert len(result_state["retrieved_chunks"]) > 0, "Should retrieve at least one chunk"
        assert result_state["retrieved_chunks"][0]["text"] != "", "Chunks should have text"
        assert "score" in result_state["retrieved_chunks"][0], "Chunks should have similarity scores"
        assert len(result_state["messages"]) >= 2, "Should log entry and completion"
        
    def test_query_agent_returns_correct_structure(self):
        
        # ARRANGE
        initial_state: AgentState = {
            "query": "test query",
            "retrieved_chunks": [],
            "retrieval_quality": "",
            "confidence_score": 0.0,
            "analysis_reason": "",
            "final_answer": "",
            "should_generate_response": False,
            "messages": [],
            "current_step": "query"
        }
        
        # ACT
        result_state = query_agent(initial_state)
        
        # ASSERT
        if result_state["retrieved_chunks"]:
            chunk = result_state["retrieved_chunks"][0]
            assert "text" in chunk
            assert "source" in chunk
            assert "score" in chunk
            assert "chunk_index" in chunk
            
class TestAnalysisAgent:
    
    def test_analysis_agent_high_confidence(self):

        # ARRANGE
        state: AgentState = {
            "query": "test",
            "retrieved_chunks": [
                {"text": "relevant content", 
                 "source": "test.txt", 
                 "score": 0.85, 
                 "chunk_index": 0}
            ],
            "retrieval_quality": "",
            "confidence_score": 0.0,
            "analysis_reason": "",
            "final_answer": "",
            "should_generate_response": False,
            "messages": [],
            "current_step": "analysis"
        }
        
        # ACT
        result_state = analysis_agent(state)
        
        # ASSERT
        assert result_state["retrieval_quality"] == "good"
        assert result_state["should_generate_response"] == True
        assert result_state["confidence_score"] >= 0.7
    
    def test_analysis_agent_low_confidence(self):
        # ARRANGE
        state: AgentState = {
            "query": "test",
            "retrieved_chunks": [
                {"text": "barely relevant",
                 "source": "test.txt",
                 "score": 0.2,
                 "chunk_index": 0}
            ],
            "retrieval_quality": "",
            "confidence_score": 0.0,
            "analysis_reason": "",
            "final_answer": "",
            "should_generate_response": False,
            "messages": [],
            "current_step": "analysis"
        }
        
        # ACT
        result_state = analysis_agent(state)
        
        # ASSERT
        assert result_state["retrieval_quality"] == "poor"
        assert result_state["should_generate_response"] == False
        assert result_state["confidence_score"] < 0.4
        assert result_state["final_answer"] != "", "Should provide direct response"
    
    def test_analysis_agent_no_results(self):
        state: AgentState = {
            "query": "test",
            "retrieved_chunks": [],
            "retrieval_quality": "",
            "confidence_score": 0.0,
            "analysis_reason": "",
            "final_answer": "",
            "should_generate_response": False,
            "messages": [],
            "current_step": "analysis"
        }
        
        result_state = analysis_agent(state)
        
        assert result_state["retrieval_quality"] == "no_results"
        assert result_state["should_generate_response"] == False
        assert result_state["confidence_score"] == 0.0


    
def test_langgraph_orchestration():
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
    
