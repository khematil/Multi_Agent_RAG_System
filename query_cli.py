from config import COLLECTION_NAME
from graph import run_rag_query


def query_rag(client, model, question: str, n_results: int = 5):

    query_vector = model.embed_query([question])[0]
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=n_results
    )
    
    return results.points

def display_results(result):
    print("\n" + "="*80)
    
    print("Answer:")
    print(f"{result['final_answer']}\n")
    
    print(f"Sources:")
    for i, chunk in enumerate(result['retrieved_chunks'][:3], 1):
        print(f"  [{i}] {chunk['source']} (chunk {chunk['chunk_index']}) - [Score] {chunk['score']:.3f}")
    
    
    print(f"\nAgent Flow:")
    for msg in result['messages']:
        print(f"{msg}") 
    
    print("=" * 80)
    
def query_cli():
    print("[RAG Query CLI -- LangGraph Orchestration]")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not question:
            continue
        
        try:
            
            result = run_rag_query(question)
            
            if result:
                display_results(result)
                
            else:
                print("No results found.")
                
        except Exception as e:
            print(f"Error: {e}")
            
            
if __name__ == "__main__":
    query_cli()
            
                                
                                
                                
                                