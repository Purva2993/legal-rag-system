# api/schemas.py
from pydantic import BaseModel
from typing import Optional, List

class IngestRequest(BaseModel):
    file_path: str

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class SourceChunk(BaseModel):
    source: str
    preview: str
    page: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    confidence: float
    guardrail_triggered: bool
    trace: dict