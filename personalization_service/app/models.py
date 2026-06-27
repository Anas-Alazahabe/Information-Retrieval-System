from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryEventRequest(BaseModel):
    user_id: str
    query_text: str


class ClickEventRequest(BaseModel):
    user_id: str
    doc_id: str
    query_text: Optional[str] = None


class RerankRequest(BaseModel):
    user_id: str
    query_text: str = ""
    results: Dict[str, float] = Field(default_factory=dict)
    alpha: Optional[float] = None


class BoostedDoc(BaseModel):
    doc_id: str
    delta_rank: int
    original_rank: int
    new_rank: int


class RerankResponse(BaseModel):
    personalization_applied: bool
    alpha: float
    profile_terms_used: List[str] = Field(default_factory=list)
    boosted_docs: List[BoostedDoc] = Field(default_factory=list)
    results: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""
    missing_doc_count: int = 0


class ProfileResponse(BaseModel):
    user_id: str
    interest_terms: Dict[str, float] = Field(default_factory=dict)
    query_count: int = 0
    click_count: int = 0


class EventResponse(BaseModel):
    status: str = "ok"
    user_id: str
    terms_added: List[str] = Field(default_factory=list)
