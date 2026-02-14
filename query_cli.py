from config import COLLECTION_NAME
from utils import get_qdrant_client, get_embedding_model

def query_rag(client, model, question: str, n_results: int = 5):

    
    query_vector = model.encode([question])[0]
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=n_results
    )
    
    return results.points

def display_results(results):
    print("\n" + "="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Similarity: {result.score:.3f}")
        print(f"Source: {result.payload.get('source_file', 'Unknown')}")
        print(f"Chunk: {result.payload.get('chunk_index', 'N/A')}")
        print(f"\n{result.payload.get('text', '')}\n")
        print("-"*80)
    
    
def main():
    print("RAG Query CLI")
    print("Type 'quit' to exit\n")
    
    client = get_qdrant_client()
    model = get_embedding_model()
    
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not question:
            continue
        
        try:
            
            results = query_rag(client, model, question, n_results = 5)
            
            if results:
                display_results(results)
                
            else:
                print("No results found.")
                
        except Exception as e:
            print(f"Error: {e}")
            
if __name__ == "__main__":
    main()
                
            
                                
                                
                                
                                