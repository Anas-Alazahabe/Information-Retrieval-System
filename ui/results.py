"""Result cards with full document content, personalization badges, and click logging."""

import html
import re
from typing import Dict, List, Optional, Set

import streamlit as st

from shared.doc_store import fetch_document_texts
from shared.search_pipeline import log_personalization_click_event
from ui.constants import CLICK_BUTTON_HELP, CLICK_BUTTON_LABEL, SNIPPET_MAX_LEN


def _tokenize_for_highlight(text: str) -> Set[str]:
    return {t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text) if len(t) >= 2}


def _highlight_terms(snippet: str, query_terms: Set[str]) -> str:
    if not query_terms:
        return html.escape(snippet)

    def replacer(match: re.Match) -> str:
        word = match.group(0)
        if word.lower() in query_terms:
            return f"<mark>{html.escape(word)}</mark>"
        return html.escape(word)

    return re.sub(r"[a-zA-Z0-9]+", replacer, snippet)


def _truncate_snippet(text: str, max_len: int = SNIPPET_MAX_LEN) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _matched_profile_terms(content: str, profile_terms: List[str]) -> List[str]:
    if not content or not profile_terms:
        return []
    doc_tokens = _tokenize_for_highlight(content)
    return [term for term in profile_terms if term.lower() in doc_tokens]


def _build_boosted_lookup(personalization_meta: Optional[dict]) -> Dict[str, dict]:
    if not personalization_meta:
        return {}
    lookup: Dict[str, dict] = {}
    for item in personalization_meta.get("boosted_docs") or []:
        doc_id = str(item.get("doc_id", ""))
        if doc_id:
            lookup[doc_id] = item
    return lookup


def render_results(
    results: Dict[str, float],
    *,
    query: str,
    refinement_meta: Optional[dict],
    personalization_meta: Optional[dict],
    use_personalization: bool,
    personalization_user_id: Optional[str],
) -> None:
    if not results:
        return

    st.markdown("### النتائج")
    st.caption(
        "مرتبة حسب درجة الصلة. معرّف الوثيقة والنص الأصلي الكامل من قاعدة البيانات."
    )

    doc_ids = list(results.keys())
    doc_texts = fetch_document_texts(doc_ids)
    found_count = sum(1 for doc_id in doc_ids if doc_texts.get(doc_id))
    st.caption(f"تغطية قاعدة البيانات: {found_count}/{len(doc_ids)} وثيقة")

    use_short_snippet = st.checkbox(
        "عرض مقتطف قصير بدلاً من النص الكامل",
        value=False,
        key="results_short_snippet",
    )

    boosted_lookup = _build_boosted_lookup(personalization_meta)
    profile_terms = (personalization_meta or {}).get("profile_terms_used") or []

    highlight_source = query
    if refinement_meta and refinement_meta.get("raw_query"):
        highlight_source = refinement_meta["raw_query"]
    query_terms = _tokenize_for_highlight(highlight_source)

    for rank, (doc_id, score) in enumerate(results.items(), 1):
        content = doc_texts.get(doc_id, "")
        boost_info = boosted_lookup.get(doc_id)
        matched = _matched_profile_terms(content, profile_terms) if boost_info else []

        badge_html = ""
        if boost_info:
            orig = boost_info.get("original_rank")
            new = boost_info.get("new_rank")
            delta = boost_info.get("delta_rank")
            badge_html = (
                f"<span class='personalized-badge'>↑ ارتفع من #{orig} إلى #{new} "
                f"(+{delta}) · Personalized re-rank</span>"
            )

        if content:
            display_text = (
                _truncate_snippet(content) if use_short_snippet else content
            )
            body_html = _highlight_terms(display_text, query_terms)
            if not use_short_snippet and len(content) > SNIPPET_MAX_LEN:
                char_note = f" ({len(content):,} حرف)"
            else:
                char_note = ""
        else:
            body_html = "<em>نص الوثيقة غير متوفر في قاعدة البيانات</em>"
            char_note = ""

        matched_html = ""
        if matched:
            matched_html = (
                f"<div class='matched-terms'>مصطلحات ملفك المطابقة: "
                f"{html.escape(', '.join(matched[:6]))}</div>"
            )

        with st.expander(
            f"#{rank} · {doc_id[:48]}{'…' if len(doc_id) > 48 else ''} · درجة {score:.4f}{char_note}",
            expanded=rank <= 3,
        ):
            st.markdown(
                f"""
<div class="result-card">
  <div class="result-card-header">
    <div>
      <span class="rank-badge">#{rank}</span>
      {badge_html}
    </div>
    <span class="score-badge">درجة الصلة: {score:.4f}</span>
  </div>
  <div class="doc-id"><strong>معرّف الوثيقة:</strong> {html.escape(doc_id)}</div>
  <div class="snippet-text">{body_html}</div>
  {matched_html}
</div>
                """,
                unsafe_allow_html=True,
            )

            if use_personalization and personalization_user_id:
                if st.button(
                    CLICK_BUTTON_LABEL,
                    key=f"click_{doc_id}_{rank}",
                    help=CLICK_BUTTON_HELP,
                ):
                    ok = log_personalization_click_event(
                        personalization_user_id,
                        doc_id,
                        query.strip(),
                    )
                    if ok:
                        st.toast(
                            "تم تحديث اهتماماتك — ستؤثر على عمليات البحث القادمة",
                            icon="✅",
                        )
                    else:
                        st.toast(
                            "تعذّر تحديث الاهتمامات — تحقق من خدمة التخصيص",
                            icon="⚠️",
                        )
