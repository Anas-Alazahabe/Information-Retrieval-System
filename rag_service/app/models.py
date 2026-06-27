from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    query: str
    results: Dict[str, float] = Field(default_factory=dict)
    top_context_docs: Optional[int] = None
    max_context_chars: Optional[int] = None
    model: Optional[str] = None
    include_citations: bool = True


class Citation(BaseModel):
    doc_id: str
    snippet: str
    retrieval_score: float


class GenerateTiming(BaseModel):
    fetch_ms: float
    generate_ms: float
    total_ms: float


class GenerateResponse(BaseModel):
    status: str = "success"
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    context_doc_ids: List[str] = Field(default_factory=list)
    missing_doc_ids: List[str] = Field(default_factory=list)
    model: str
    timing: GenerateTiming
