"""Main search input, suggestions, and search trigger."""

from typing import List, Optional

import requests
import streamlit as st
from st_keyup import st_keyup

from ui.constants import SUGGEST_DEFAULT_LIMIT, SUGGEST_MIN_PREFIX_LEN, SUGGESTION_DISPLAY_MAX_LEN


def truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def fetch_suggestions(suggest_service_url: str, prefix: str) -> List[str]:
    if len(prefix.strip()) < SUGGEST_MIN_PREFIX_LEN:
        return []
    try:
        response = requests.get(
            suggest_service_url,
            params={"q": prefix, "limit": SUGGEST_DEFAULT_LIMIT},
            timeout=2,
        )
        if response.status_code == 200:
            return response.json().get("suggestions", [])
    except Exception:
        pass
    return []


def render_search_input() -> tuple[str, bool]:
    """Render query box, suggestions, and search button. Returns (query, should_search)."""
    st.markdown("### البحث")
    st.caption(
        "ماذا يفعل هذا؟ تكتب سؤالك بلغة طبيعية. "
        "لماذا؟ محرك البحث يجد المقاطع الأنسب من ملايين الوثائق."
    )

    query = st_keyup(
        "أدخل استعلامك:",
        value=st.session_state.get("query", ""),
        key="query",
        debounce=300,
        placeholder="مثال: hospital patient care أو how to learn python...",
    ) or ""

    suggest_url: Optional[str] = st.session_state.get("suggest_url")
    suggestions: List[str] = []
    if suggest_url:
        suggestions = fetch_suggestions(suggest_url, query)

    if query.strip() and len(query.strip()) >= SUGGEST_MIN_PREFIX_LEN and suggestions:
        st.caption("اقتراحات — استعلامات شائعة من مجموعة MS MARCO (Query Suggestions)")
        cols = st.columns(min(len(suggestions), 3))
        for index, suggestion in enumerate(suggestions[:6]):
            label = truncate(suggestion, SUGGESTION_DISPLAY_MAX_LEN)
            col = cols[index % len(cols)]
            if col.button(
                label,
                key=f"suggest_{index}_{abs(hash(suggestion)) % 10_000}",
                help=suggestion,
                use_container_width=True,
            ):
                st.session_state.pending_query = suggestion
                st.session_state.trigger_search = True
                st.rerun()

    if st.button("ابحث الآن", type="primary", use_container_width=False):
        st.session_state.trigger_search = True

    should_search = st.session_state.pop("trigger_search", False)
    return query, should_search
