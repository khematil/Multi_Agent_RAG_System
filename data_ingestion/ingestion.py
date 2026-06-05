from pathlib import Path
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams, PointStruct
import argparse

from config import *
from utils import get_qdrant_client, get_embedding_model

def ingest_document(file_path: str):
    """
    Function that ingests documents located in a file path / folder

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

def process_single_document(file_path: str) -> tuple:
    """
    Process a single file: load, chunk, and add metadata.
    
    Args:
        file_path: Path to file
        
    Returns:
        Tuple of (chunks, stats_dict)
    """
    
    filename = Path(file_path).name
    try:
        # 1. Load document
        print(f"Processing: {filename}")
        document = ingest_document(file_path)
        
        # 2. Chunk document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(document)
        print(f"\tCreated {len(chunks)} chunks")
        
        # 3. Add metadata for logging
        for i, chunk in enumerate(chunks):
            chunk.metadata['source_file'] = filename
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks, {
            "success": True,
            "filename": filename,
            "chunks_created": len(chunks)   
        }
    except Exception as e:
        print(f"Error: {e}")
        return [], {
            'success': False,
            'filename': filename,
            'error': str(e)
        }

def process_documents(directory_path: str, skip_existing: bool = True):
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
        'files_already_exists': 0,
        'total_chunks': 0,
        'by_file': {}
    }
    
    dir = Path(directory_path)
    extensions = ['.csv', '.txt', '.pdf']
    
    
    existing_files = get_existing_files_in_qdrant() if skip_existing else set()
    if existing_files:
        print(f"Found {len(existing_files)} files already in Qdrant\n")
        print(f"Will skip: {', '.join(sorted(existing_files))}\n")
    
    for file_path in dir.iterdir():
        if not file_path.is_file():
            continue
        
        if file_path.suffix not in extensions:
            print(f"Skipping unsupported file: {file_path.name}")
            stats['files_skipped'] += 1
            continue

        if skip_existing and file_path.name in existing_files:
            print(f"Skipping (already exists): {file_path.name}")
            stats['files_already_exists'] += 1
            continue
        
        chunks, file_stats = process_single_document(str(file_path))
        
        
        if file_stats["success"]:
            all_chunks.extend(chunks)
            stats["files_processed"] += 1
            stats["by_file"][file_path.name] = len(chunks)
        else:
            stats["files_failed"] += 1
            
    stats['total_chunks'] = len(all_chunks)
    
    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"\tFiles processed: {stats['files_processed']}")
    print(f"\tFiles already existed: {stats['files_already_exists']}")
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
        existing_count = collection_info.points_count
        print(f"\tCollection '{COLLECTION_NAME}' exists ({existing_count} existing points)")
        
    except:
        print(f"\t Creating new collection '{COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
        existing_count = 0
    
    start_id = existing_count
    
    print("\tGenerating embeddings")
    texts = [chunk.page_content for chunk in chunks]
    # embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = model.embed_documents(texts) # LangChain API use
        
    print("\tCreating points...")
    points = []
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=start_id + i, 
            vector=embedding,
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

def get_existing_files_in_qdrant() -> set:
    """
    Get set of filenames already in Qdrant.
    
    Returns:
        Set of filenames (e.g., {'file1.txt', 'file2.pdf'})
    """
    try:
        client = get_qdrant_client()
        
        existing_files = set()
        offset = None
        
        while True:
            records, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False  
            )
            
            for record in records:
                source_file = record.payload.get('source_file')
                if source_file:
                    existing_files.add(source_file)
            
            if next_offset is None:
                break
            offset = next_offset
        
        return existing_files
        
    except Exception as e:
        print(f"Warning: Could not check existing files: {e}")
        return set()  


def clear_collection():
    """Delete the entire Qdrant collection."""
    
    print("="*80)
    print("WARNING: CLEAR QDRANT COLLECTION")
    print("="*80)
    print(f"\nThis will DELETE all data in '{COLLECTION_NAME}'")
    print("This action CANNOT be undone!\n")
    
    # 2x Confirm
    confirm1 = input("Type 'DELETE' to confirm: ")
    if confirm1 != "DELETE":
        print("Cancelled")
        return
    
    confirm2 = input("Are you absolutely sure? (yes/no): ")
    if confirm2.lower() != "yes":
        print("Cancelled")
        return
    
    try:
        client = get_qdrant_client()
        info = client.get_collection(COLLECTION_NAME)
        
        print(f"\nDeleting collection with {info.points_count} points...")
        client.delete_collection(COLLECTION_NAME)
        print("Collection deleted successfully")
        
    except Exception as e:
        print(f"Error: {e}")





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
    clear_collection()
    main()