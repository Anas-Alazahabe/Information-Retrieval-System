from typing import Dict, List

from pydantic import BaseModel, Field


class RefineRequest(BaseModel):
    """طلب تحسين استعلام المستخدم."""

    raw_query: str
    enabled_techniques: List[str] = Field(default_factory=list)
    previous_queries: List[str] = Field(default_factory=list)
    representation_mode: str = "bm25"


class RefineResponse(BaseModel):
    """نتيجة تحسين الاستعلام مع تلميحات للمعالجة اللاحقة."""

    raw_query: str
    refined_query: str
    expanded_terms: List[str] = Field(default_factory=list)
    techniques_applied: List[str] = Field(default_factory=list)
    explanation: str = ""
    preprocess_hints: Dict[str, bool] = Field(default_factory=dict)


class SuggestResponse(BaseModel):
    """اقتراحات استعلام بناءً على بادئة نصية."""

    query_prefix: str
    suggestions: List[str] = Field(default_factory=list)
