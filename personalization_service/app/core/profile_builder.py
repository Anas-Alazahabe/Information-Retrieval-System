"""Build user interest term weights from query and click events."""

from typing import Dict

from .tokenizer import lightweight_tokenize
from shared.ir_config import CLICK_EVENT_WEIGHT, QUERY_EVENT_WEIGHT


def terms_from_text(text: str, weight: float) -> Dict[str, float]:
    terms: Dict[str, float] = {}
    for token in lightweight_tokenize(text):
        terms[token] = terms.get(token, 0.0) + weight
    return terms


def terms_from_query(query_text: str) -> Dict[str, float]:
    return terms_from_text(query_text, QUERY_EVENT_WEIGHT)


def terms_from_click(doc_content: str) -> Dict[str, float]:
    return terms_from_text(doc_content, CLICK_EVENT_WEIGHT)
