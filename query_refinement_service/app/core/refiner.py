import logging
import re
from typing import Dict, List, Tuple

import requests

from app.core.history import merge_history_terms
from app.core.prf import prf_rm3
from app.core.synonym_expander import expand_synonyms
from shared.ir_config import (
    PREPROCESS_FLAGS,
    VALID_REFINEMENT_TECHNIQUES,
    WH_WORDS,
    preprocess_single_url,
)

logger = logging.getLogger(__name__)


def _normalize_query(text: str) -> str:
    """تطبيع بسيط للنص: trim وتقليص المسافات."""
    return re.sub(r"\s+", " ", text.strip())


def _merge_query_string(base_query: str, new_terms: List[str]) -> str:
    """Append unique terms to the query string."""
    if not new_terms:
        return base_query
    existing = set(base_query.lower().split())
    additions = [t for t in new_terms if t.lower() not in existing]
    if not additions:
        return base_query
    return f"{base_query} {' '.join(additions)}"


def is_question_query(text: str) -> bool:
    """يكشف الاستعلامات على شكل سؤال (WH-word أو علامة استفهام)."""
    normalized = _normalize_query(text).lower()
    if not normalized:
        return False
    if normalized.endswith("?"):
        return True
    first_token = normalized.split()[0]
    return first_token in WH_WORDS


def apply_query_preprocess(raw_query: str) -> Tuple[str, Dict[str, bool], str]:
    """يطبّق قواعد المعالجة الخاصة بالاستعلام ويعيد تلميحات preprocessing."""
    refined = _normalize_query(raw_query)
    if is_question_query(refined):
        hints = {"preserve_wh_words": True}
        explanation = (
            "Question-style query detected; WH-words (what/how/why/...) "
            "will be preserved during stopword removal."
        )
    else:
        hints = {}
        explanation = "Non-question query; standard document-style preprocessing applies."
    return refined, hints, explanation


class QueryRefiner:
    """منسّق تقنيات تحسين الاستعلام."""

    def _fetch_query_tokens(self, query: str, preserve_wh_words: bool) -> List[str]:
        """Tokenize query via preprocessing service."""
        payload = {
            "text": query,
            **PREPROCESS_FLAGS,
            "preserve_wh_words": preserve_wh_words,
        }
        try:
            response = requests.post(preprocess_single_url(), json=payload, timeout=10)
            response.raise_for_status()
            return response.json().get("tokens", [])
        except Exception as exc:
            logger.warning("Preprocessing unavailable for refinement: %s", exc)
            return query.lower().split()

    def refine(
        self,
        raw_query: str,
        enabled_techniques: List[str],
        previous_queries: List[str] | None = None,
        representation_mode: str = "bm25",
    ) -> dict:
        """ينفّذ التقنيات المفعّلة ويعيد الاستعلام المحسّن مع البيانات الوصفية."""
        del representation_mode  # PRF first pass always uses BM25 internally

        if not raw_query or not raw_query.strip():
            raise ValueError("Query text is empty.")

        techniques = [
            t for t in enabled_techniques if t in VALID_REFINEMENT_TECHNIQUES
        ]
        refined_query = _normalize_query(raw_query)
        expanded_terms: List[str] = []
        techniques_applied: List[str] = []
        preprocess_hints: Dict[str, bool] = {}
        explanation_parts: List[str] = []
        working_tokens: List[str] = []

        if "query_preprocess" in techniques:
            refined_query, hints, fragment = apply_query_preprocess(raw_query)
            preprocess_hints.update(hints)
            techniques_applied.append("query_preprocess")
            explanation_parts.append(fragment)

        preserve_wh = preprocess_hints.get("preserve_wh_words", False)

        if "history" in techniques:
            history_terms, history_explanation = merge_history_terms(
                current_query=refined_query,
                current_tokens=working_tokens,
                previous_queries=previous_queries or [],
            )
            if history_terms:
                expanded_terms.extend(history_terms)
                working_tokens.extend(history_terms)
                refined_query = _merge_query_string(refined_query, history_terms)
            explanation_parts.append(history_explanation)
            techniques_applied.append("history")

        needs_tokens = "synonyms" in techniques or "prf" in techniques
        if needs_tokens and not working_tokens:
            working_tokens = self._fetch_query_tokens(refined_query, preserve_wh)

        if "synonyms" in techniques:
            synonym_terms = expand_synonyms(working_tokens, skip_terms=WH_WORDS)
            if synonym_terms:
                expanded_terms.extend(synonym_terms)
                working_tokens.extend(synonym_terms)
                refined_query = _merge_query_string(refined_query, synonym_terms)
                explanation_parts.append(f"WordNet added: {', '.join(synonym_terms)}.")
            else:
                explanation_parts.append("WordNet: no synonyms added.")
            techniques_applied.append("synonyms")

        if "prf" in techniques:
            prf_terms, prf_explanation = prf_rm3(
                query_tokens=working_tokens or self._fetch_query_tokens(
                    refined_query, preserve_wh
                ),
                preserve_wh_words=preserve_wh,
            )
            if prf_terms:
                expanded_terms.extend(prf_terms)
                working_tokens.extend(prf_terms)
                refined_query = _merge_query_string(refined_query, prf_terms)
            explanation_parts.append(prf_explanation)
            techniques_applied.append("prf")

        if not techniques_applied:
            explanation = "No refinement techniques enabled."
        else:
            explanation = " ".join(explanation_parts)

        return {
            "raw_query": raw_query,
            "refined_query": refined_query,
            "expanded_terms": expanded_terms,
            "techniques_applied": techniques_applied,
            "explanation": explanation,
            "preprocess_hints": preprocess_hints,
        }
