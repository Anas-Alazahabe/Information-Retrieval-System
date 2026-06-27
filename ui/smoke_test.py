"""Quick end-to-end smoke test from the UI."""

from typing import Any, Dict, List, Tuple

import requests
import streamlit as st

from shared.ir_config import RETRIEVAL_URL


def run_smoke_test(
    *,
    retrieval_url: str = RETRIEVAL_URL,
    query: str = "hospital information system",
) -> List[Tuple[str, bool, str]]:
    """Run BM25 + embedding searches; return (label, ok, detail) rows."""
    base = retrieval_url.rstrip("/")
    modes = [
        ("BM25", "bm25"),
        ("Embedding", "embedding"),
    ]
    rows: List[Tuple[str, bool, str]] = []
    for label, mode in modes:
        try:
            response = requests.post(
                f"{base}/search",
                json={
                    "query": query,
                    "representation_mode": mode,
                    "top_k": 5,
                },
                timeout=60,
            )
            if response.status_code != 200:
                rows.append((label, False, f"HTTP {response.status_code}"))
                continue
            data: Dict[str, Any] = response.json()
            count = len(data.get("results") or {})
            ok = data.get("status") == "success" and count > 0
            rows.append((label, ok, f"{count} نتائج · {data.get('total_ms', data.get('timing', {}).get('total_ms', '?'))} ms"))
        except Exception as exc:
            rows.append((label, False, str(exc)))
    return rows


def render_smoke_test_button(*, retrieval_url: str = RETRIEVAL_URL) -> None:
    if st.button("اختبار سريع للخدمات", help="يبحث باستعلام ثابت عبر BM25 و Embedding"):
        with st.spinner("جاري الاختبار..."):
            rows = run_smoke_test(retrieval_url=retrieval_url)
        for label, ok, detail in rows:
            if ok:
                st.success(f"{label}: نجح — {detail}")
            else:
                st.error(f"{label}: فشل — {detail}")
