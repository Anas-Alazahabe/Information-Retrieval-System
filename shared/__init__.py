"""Shared configuration and indexing utilities for IR microservices."""

from shared.ir_config import (
    PERSONALIZATION_ALPHA,
    PERSONALIZATION_RERANK_POOL,
    PERSONALIZATION_URL,
    REFINEMENT_URL,
    RETRIEVAL_URL,
    personalize_click_event_url,
    personalize_profile_url,
    personalize_query_event_url,
    personalize_rerank_url,
)

__all__ = [
    "PERSONALIZATION_ALPHA",
    "PERSONALIZATION_RERANK_POOL",
    "PERSONALIZATION_URL",
    "REFINEMENT_URL",
    "RETRIEVAL_URL",
    "personalize_click_event_url",
    "personalize_profile_url",
    "personalize_query_event_url",
    "personalize_rerank_url",
]
