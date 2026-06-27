"""Post-search summary strip and technical details expander."""

from typing import Any, Dict, Optional

import streamlit as st

from ui.constants import (
    ENHANCED_QUERY_SUMMARY_MAX_LEN,
    REFINEMENT_TECHNIQUE_LABELS,
    retrieval_mode_label,
)


def truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def format_timing(timing: dict) -> str:
    if not timing:
        return ""
    parts = []
    for key, label in (
        ("preprocess_ms", "معالجة"),
        ("encode_ms", "تضمين"),
        ("match_ms", "مطابقة"),
        ("rank_ms", "ترتيب"),
        ("ranking_ms", "ترتيب"),
        ("total_ms", "الإجمالي"),
    ):
        if key in timing:
            parts.append(f"{label}: {timing[key]:.1f} ms")
    return " | ".join(parts)


def render_summary_strip(
    data: Dict[str, Any],
    refinement_meta: Optional[dict],
    personalization_meta: Optional[dict],
    *,
    use_refinement: bool,
    use_personalization: bool,
    rag_meta: Optional[dict] = None,
    use_rag: bool = False,
) -> None:
    chips: list[str] = []

    mode_used = data.get("mode_used", "")
    if mode_used:
        chips.append(
            f"<span class='summary-chip'><strong>الاسترجاع:</strong> "
            f"{retrieval_mode_label(mode_used)}</span>"
        )

    if use_refinement and refinement_meta:
        raw_q = refinement_meta.get("raw_query", "")
        refined_q = refinement_meta.get("refined_query", "")
        if raw_q and refined_q and raw_q != refined_q:
            chips.append(
                f"<span class='summary-chip'><strong>تحسين:</strong> نعم — "
                f"{truncate(raw_q, 40)} → {truncate(refined_q, ENHANCED_QUERY_SUMMARY_MAX_LEN)}</span>"
            )
        else:
            chips.append(
                "<span class='summary-chip'><strong>تحسين:</strong> "
                "لم يتغير الاستعلام</span>"
            )
    elif not use_refinement:
        chips.append(
            "<span class='summary-chip'><strong>تحسين:</strong> معطّل (بحث مباشر)</span>"
        )

    if use_personalization:
        if personalization_meta and personalization_meta.get("personalization_applied"):
            boosted_count = len(personalization_meta.get("boosted_docs") or [])
            terms = personalization_meta.get("profile_terms_used") or []
            term_preview = ", ".join(terms[:3]) if terms else "—"
            chips.append(
                f"<span class='summary-chip'><strong>تخصيص:</strong> مفعّل — "
                f"{boosted_count} وثيقة أعيد ترتيبها · {term_preview}</span>"
            )
        elif personalization_meta:
            explanation = personalization_meta.get("explanation", "غير مطبّق")
            chips.append(
                f"<span class='summary-chip'><strong>تخصيص:</strong> {truncate(explanation, 60)}</span>"
            )
        else:
            chips.append(
                "<span class='summary-chip'><strong>تخصيص:</strong> غير متاح</span>"
            )
    else:
        chips.append(
            "<span class='summary-chip'><strong>تخصيص:</strong> معطّل</span>"
        )

    if use_rag:
        if rag_meta and rag_meta.get("answer"):
            model = rag_meta.get("model", "")
            ctx_count = len(rag_meta.get("context_doc_ids") or [])
            gen_ms = (rag_meta.get("timing") or {}).get("generate_ms")
            gen_label = f" · {gen_ms:.0f} ms" if gen_ms is not None else ""
            chips.append(
                f"<span class='summary-chip'><strong>RAG:</strong> مفعّل — "
                f"{ctx_count} مقطع · {model}{gen_label}</span>"
            )
        else:
            chips.append(
                "<span class='summary-chip'><strong>RAG:</strong> غير متاح</span>"
            )
    else:
        chips.append(
            "<span class='summary-chip'><strong>RAG:</strong> معطّل</span>"
        )

    timing = data.get("timing") or {}
    total_ms = timing.get("total_ms")
    if total_ms is not None:
        chips.append(
            f"<span class='summary-chip'><strong>الزمن:</strong> {total_ms:.0f} ms</span>"
        )

    if chips:
        st.markdown(
            f"<div class='summary-strip'>{''.join(chips)}</div>",
            unsafe_allow_html=True,
        )


def _render_enhancement_details(refinement_meta: dict) -> None:
    st.markdown("#### تحسين الاستعلام")
    col_raw, col_refined = st.columns(2)
    col_raw.markdown(f"**استعلامك:** {refinement_meta.get('raw_query', '')}")
    col_refined.markdown(f"**الاستعلام المحسّن:** {refinement_meta.get('refined_query', '')}")

    if refinement_meta.get("techniques_applied"):
        labels = [
            REFINEMENT_TECHNIQUE_LABELS.get(t, t)
            for t in refinement_meta["techniques_applied"]
        ]
        st.caption(f"التحسينات المطبّقة: {', '.join(labels)}")

    explanation = refinement_meta.get("explanation", "")
    expanded_terms = refinement_meta.get("expanded_terms") or []
    if explanation:
        st.info(explanation)
    elif expanded_terms:
        st.caption(f"مصطلحات مضافة: {', '.join(expanded_terms)}")


def _render_personalization_details(personalization_meta: dict) -> None:
    st.markdown("#### التخصيص")
    st.markdown(
        f"**تخصيص مفعّل:** {personalization_meta.get('personalization_applied', False)}"
    )
    st.markdown(f"**alpha:** {personalization_meta.get('alpha')}")
    explanation = personalization_meta.get("explanation", "")
    if explanation:
        st.info(explanation)
    boosted = personalization_meta.get("boosted_docs") or []
    if boosted:
        st.caption("وثائق تمت إعادة ترتيبها للأعلى:")
        for item in boosted[:10]:
            st.caption(
                f"- {item.get('doc_id')}: "
                f"#{item.get('original_rank')} → #{item.get('new_rank')} "
                f"(+{item.get('delta_rank')})"
            )


def _render_rag_details(rag_meta: dict) -> None:
    st.markdown("#### الإجابة الذكية (RAG)")
    st.markdown(f"**model:** {rag_meta.get('model', '')}")
    st.markdown(f"**answer:** {rag_meta.get('answer', '')}")
    context_ids = rag_meta.get("context_doc_ids") or []
    if context_ids:
        st.caption(f"context_doc_ids: {', '.join(context_ids)}")
    timing = rag_meta.get("timing") or {}
    if timing:
        st.caption(
            f"fetch_ms={timing.get('fetch_ms')} · "
            f"generate_ms={timing.get('generate_ms')} · "
            f"total_ms={timing.get('total_ms')}"
        )


def render_technical_details(
    data: Dict[str, Any],
    refinement_meta: Optional[dict],
    personalization_meta: Optional[dict],
    rag_meta: Optional[dict] = None,
) -> None:
    with st.expander("تفاصيل تقنية (للعرض الأكاديمي)", expanded=False):
        if refinement_meta:
            _render_enhancement_details(refinement_meta)
        if personalization_meta:
            _render_personalization_details(personalization_meta)
        if rag_meta:
            _render_rag_details(rag_meta)

        matching_method = data.get("matching_method")
        if matching_method:
            st.caption(f"matching_method: {matching_method}")
        params = data.get("params")
        if params:
            st.caption(f"params: {params}")

        st.caption(f"query_tokens: {data.get('query_tokens', [])}")
        timing_text = format_timing(data.get("timing", {}))
        if timing_text:
            st.caption(f"timing: {timing_text}")
