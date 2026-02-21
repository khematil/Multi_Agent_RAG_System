from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone

class Source(BaseModel):
    source: str = Field(
        ...,
        description="Source filename or document id",
        example="sample_data_distributed_systems.txt"
    )
    chunk_index: int = Field(
        ...,
        description="Index of chunk in source document",
        example=5
    )
    score: float = Field(
        ...,
        description="Cosine similarity score (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        example=0.892
    )
    class Config:
        json_schema_extra = {
            "example": {
                "source": "sample_data_distributed_systems.txt",
                "chunk_index": 3,
                "score": 0.892
            }
        }

class QueryRequest(BaseModel):

    question: str = Field(
        ..., 
        description="The question to ask the RAG system",
        min_length=1,
        max_length=1000,
        example="What is the CAP theorem?"
    )
    max_results: int = Field(
        default=5,
        description="Max number of source chunks to retrieve",
        ge=1,
        le=20,
        example=5
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source reference in response",
        example=True
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the CAP theorem?",
                "max_results": 5,
                "include_sources": True
            }
        }
        
class QueryResponse(BaseModel):
    
    answer: str = Field(
        ...,
        description="Answer generated from RAG system",
        example="The CAP theorem, proposed by Eric Brewer, states that a distributed system can have at most two of: Consistency, Availability, and Partition tolerance."
    )
    confidence_score: float = Field(
        ...,
        description="Confidence score of retrieval (0.0 - 1.0-)",
        ge=0.0,
        le=1.0,
        example=0.80
    )
    retrieval_quality: str = Field(
        ...,
        description="Quality assessment of retrieval",
        example="good"
    )
    sources: Optional[List[Source]] = Field(
        default=None,
        description="List of source documents used (if include_sources = True)",
        example=[
            {
                "source": "sample_data_distributed_systems.txt",
                "chunk_index": 3,
                "score": 0.80
            }
        ]
    )
    processing_time: float = Field(
        ...,
        description="Time taken to process the query in seconds",
        ge=0.0,
        example=2.0
    )
    timestamp: datetime = Field(
        default_factory= lambda: datetime.now(timezone.utc),
        description="When the query was processed (UTC)",
        example="2026-02-20T16:30:00Z"
    )
    class Config:
        """Pydantic config for better JSON schema."""
        json_schema_extra = {
            "example": {
                "answer": "The CAP theorem, proposed by Eric Brewer, states that a distributed system can have at most two of: Consistency, Availability, and Partition tolerance.",
                "confidence_score": 0.892,
                "retrieval_quality": "good",
                "sources": [
                    {
                        "source": "sample_data_distributed_systems.txt",
                        "chunk_index": 3,
                        "score": 0.892
                    },
                    {
                        "source": "sample_data_distributed_systems.txt",
                        "chunk_index": 4,
                        "score": 0.854
                    }
                ],
                "processing_time": 2.34,
                "timestamp": "2026-02-20T16:30:00Z"
            }
        }

class ErrorResponse(BaseModel):
    detail: str = Field(
        ...,
        description="Error message",
        example="Invalid request: question cannot be empty"
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable error code",
        example="INVALID_QUESTION"
    )
    timestamp: datetime = Field(
        default_factory= lambda: datetime.now(timezone.utc),
        description="When the error occurred (UTC)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Service temporarily unavailable",
                "error_code": "SERVICE_UNAVAILABLE",
                "timestamp": "2026-02-20T16:30:00Z"
            }
        }