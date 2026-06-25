from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
from typing import List

from data_ingestion.ingestion import (
    get_existing_files_in_qdrant,
    process_single_document,    
    store_in_qdrant
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME
from utils import get_qdrant_client


router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)

@router.get("/list")
async def list_documents():
    """
    List all documents in the system.
    """
    try:

        existing_files = get_existing_files_in_qdrant()
        
        client = get_qdrant_client()
        
        try:
            collection_info = client.get_collection(COLLECTION_NAME)
            total_chunks = collection_info.points_count
        except:
            total_chunks = 0
        
        return {
            "total_chunks": total_chunks,
            "unique_documents": len(existing_files),
            "documents": sorted(list(existing_files))
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )
        
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    force: bool = False
):
    """
    Upload and ingest a single document.
    
    Args:
        file: Uploaded file (txt/pdf/csv)
        force: If True, reprocess even if file exists (default: False)
    
    Returns:
        Ingestion results
    """
    allowed_extensions = {".txt", ".pdf", ".csv"}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}"
        )  
        
    try:
        if not force:
            existing_files = get_existing_files_in_qdrant()
            if file.filename in existing_files:
                raise HTTPException(
                    status_code=409,
                    detail=f"File '{file.filename} already exists. Set force=true to overwrite."
                )
        


        upload_dir = Path(DATA_DIR)
        file_path = upload_dir / file.filename


        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        chunks, file_stats = process_single_document(file_path)
        
        if not file_stats['success']:
            raise Exception(file_stats.get('error', 'Processing failed'))       
        
    
        store_in_qdrant(chunks)
        
        if file_path.exists():
            file_path.unlink()
            
        client = get_qdrant_client()
        final_count = client.get_collection(COLLECTION_NAME).points_count
        total_docs = len(get_existing_files_in_qdrant())
        return {
            "status": "success",
            "filename": file_stats['filename'],
            "chunks_created": file_stats['chunks_created'],
            "total_points": final_count,
            "total_docs": total_docs,
            "message": f"Successfully ingested {file_stats['filename']}"
        }     
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )
        

@router.delete("/{filename}")
async def delete_document(filename: str):
    """
    Delete all chunks from a specific document.
    
    Args:
        filename: Name of file to delete
    
    Returns:
        Deletion results
    """
    try:
        
        client = get_qdrant_client()
            # Find all points with this source_file
        points_to_delete = []
        offset = None
        
        print(f"Searching for chunks from: {filename}")
        
        while True:
            records, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            for record in records:
                if record.payload.get('source_file') == filename:
                    points_to_delete.append(record.id)
            
            if next_offset is None:
                break
            offset = next_offset
        
        if not points_to_delete:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{filename}' not found in system"
            )
        
        print(f"\tDeleting {len(points_to_delete)} chunks...")
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=points_to_delete
        )
        
        file_path = Path(DATA_DIR) / filename
        if file_path.exists():
            file_path.unlink()
            print(f"   Deleted file from disk: {filename}")
        
        return {
            "status": "success",
            "filename": filename,
            "chunks_deleted": len(points_to_delete),
            "message": f"Deleted {len(points_to_delete)} chunks from {filename}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )
