"""RAG answer card with citations and technical trace."""

from typing import Optional

import streamlit as st


def render_rag_answer(rag_meta: Optional[dict], *, use_rag: bool) -> None:
    if not use_rag:
        return
    if not rag_meta:
        st.warning("لم تُولَّد إجابة — تحقق من خدمة RAG ومفتاح Gemini.")
        return

    st.markdown("### الإجابة الذكية")
    st.caption("ملخص مبني على المقاطع المسترجعة — مع استشهادات بالوثائق المصدر.")

    answer = rag_meta.get("answer", "")
    if answer:
        st.markdown(answer)
    else:
        st.info("لم يُرجع النموذج نصاً.")

    citations = rag_meta.get("citations") or []
    if citations:
        st.markdown("**المصادر:**")
        for item in citations:
            doc_id = item.get("doc_id", "")
            score = item.get("retrieval_score")
            snippet = item.get("snippet", "")
            score_label = f" · score={score:.2f}" if score is not None else ""
            st.caption(f"**[{doc_id}]**{score_label}")
            if snippet:
                st.caption(snippet)

    timing = rag_meta.get("timing") or {}
    model = rag_meta.get("model", "")
    missing = rag_meta.get("missing_doc_ids") or []
    context_ids = rag_meta.get("context_doc_ids") or []

    with st.expander("تفاصيل RAG", expanded=False):
        if model:
            st.caption(f"model: {model}")
        if context_ids:
            st.caption(f"context_doc_ids: {', '.join(context_ids)}")
        if missing:
            st.caption(f"missing_doc_ids: {', '.join(missing)}")
        if timing:
            parts = []
            for key, label in (
                ("fetch_ms", "جلب النصوص"),
                ("generate_ms", "توليد Gemini"),
                ("total_ms", "الإجمالي"),
            ):
                if key in timing:
                    parts.append(f"{label}: {timing[key]:.1f} ms")
            if parts:
                st.caption(" | ".join(parts))
