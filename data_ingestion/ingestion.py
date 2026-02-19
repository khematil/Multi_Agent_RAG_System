"""
Run once to ingest data and store it into Qdrant store

"""

from pathlib import Path
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams, PointStruct
import argparse

from config import *
from utils import get_qdrant_client, get_embedding_model

def ingest_document(file_path: str):
    """Function that ingests documents located in a file path / folder

    Args:
        file_path (str): path of where data files are located

    Raises:
        ValueError: Unsupported file type

    Returns:
        list[Document]: list of LangChain Document classes
    """
    path = Path(file_path)
    
    if path.suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix == ".txt":
        loader = TextLoader(str(path))
    elif path.suffix == ".csv":
        loader = CSVLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    documents = loader.load()
    return documents

def process_documents(directory_path: str):
    """
    Process all documents in a directory and return chunked texts.
    
    Args:
        directory_path: Path to directory containing documents
        
    Returns:
        Tuple of (all_chunks, stats)
    """
    all_chunks = []
    
    stats = {
        'files_processed': 0,
        'files_failed': 0,
        'files_skipped': 0,
        'total_chunks': 0,
        'by_file': {}
    }
    
    dir = Path(directory_path)
    extensions = ['.csv', '.txt', '.pdf']
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    for file_path in dir.iterdir():
        if not file_path.is_file():
            continue
        
        if file_path.suffix not in extensions:
            print(f"Skipping unsupported file: {file_path.name}")
            stats['files_skipped'] += 1
            continue
            
        try:
            print("Processing:", file_path.name)
            
            documents = ingest_document(str(file_path))
            print(f"\tLoaded {len(documents)} document(s)")
            
            chunks = text_splitter.split_documents(documents)
            print(f"\tCreated {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                chunk.metadata['source_file'] = file_path.name
                chunk.metadata['chunk_index'] = i
                chunk.metadata['total_chunks'] = len(chunks)
                
            all_chunks.extend(chunks)
            stats['files_processed'] += 1
            stats['by_file'][file_path.name] = len(chunks)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            stats['files_failed'] += 1 
            continue
    stats['total_chunks'] = len(all_chunks)
    
    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"\tFiles processed: {stats['files_processed']}")
    print(f"\tFiles failed: {stats['files_failed']}")
    print(f"\tFiles skipped: {stats['files_skipped']}")
    print(f"\tTotal chunks: {stats['total_chunks']}")
    print(f"{'='*60}")
    
    return all_chunks, stats

def store_in_qdrant(chunks: list):
    
    if not chunks:
        print("No chunks to store in qdrant.")
        return
    
    print(f"\nStoring {len(chunks)} chunks in Qdrant")
    
    client = get_qdrant_client()
    model = get_embedding_model()
    
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"\tCollection '{COLLECTION_NAME}' exists ({collection_info.points_count} existing points)")
        
        response = input("\tClear existing data? (y/n): ").lower()
        if response == 'y':
            client.delete_collection(COLLECTION_NAME)
            print("\tDeleted old collection")
            raise Exception("Recreate")
    except:
        print(f"\t Creating new collection '{COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
    
    print("\tGenerating embeddings")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        start_id = collection_info.points_count
    except:
        start_id = 0
        
    print("\tCreating points...")
    points = []
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=start_id + i, 
            vector=embedding.tolist(),
            payload={
                'text': chunk.page_content,
                'source_file': chunk.metadata.get('source_file', 'unknown'),
                'chunk_index': chunk.metadata.get('chunk_index', 0),
                'total_chunks': chunk.metadata.get('total_chunks', 0)
            }
        ))
    
    print("\tUploading to Qdrant . . .")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    
    print(f"\tSuccessfully stored {len(points)} chunks")
    
    final_count = client.get_collection(COLLECTION_NAME).points_count
    print(f"\tTotal points in collection: {final_count}")

def main():
    
    parser = argparse.ArgumentParser(description='Ingest documents into RAG system')
    parser.add_argument('--dir', default=DATA_DIR, help='Directory containing documents')
    args = parser.parse_args()
    
    print("="*80)
    print("DOCUMENT INGESTION")
    print("="*80)
    print(f"\nProcessing directory:{args.dir}\n")
    
    chunks, stats = process_documents(args.dir)
    
    if chunks:
        store_in_qdrant(chunks)
    else:
        print("\nNo chunks to store . . . Check your data directory.")
        return

    print("\n" + "="*80)
    print("INGESTION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()