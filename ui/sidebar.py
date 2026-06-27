"""Sidebar controls: retrieval, refinement, personalization, advanced settings."""

from dataclasses import dataclass
from typing import List, Optional

import requests
import streamlit as st

from ui.constants import (
    DEFAULT_DISPLAY_TOP_K,
    DEFAULT_PERSONALIZATION_USER,
    DEFAULT_REFINEMENT_PRESET,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_RAG_CONTEXT_DOCS,
    DEFAULT_SEARCH_MODE,
    PERSONALIZATION_SECTION,
    PERSONALIZATION_USER_HINTS,
    PERSONALIZATION_USER_OPTIONS,
    RAG_CONTEXT_DOC_OPTIONS,
    RAG_SECTION,
    REFINEMENT_PRESET_HINTS,
    REFINEMENT_PRESETS,
    REFINEMENT_SECTION,
    REFINEMENT_TECHNIQUE_LABELS,
    REFINEMENT_TECHNIQUE_OPTIONS,
    RETRIEVAL_MODE_BY_KEY,
    RETRIEVAL_MODE_KEYS,
    RETRIEVAL_SECTION,
    SEARCH_MODE_COPY,
    SEARCH_MODES,
    SERIAL_HYBRID_TOP_N,
)

import shared.ir_config as ir_config


@dataclass
class SearchSettings:
    representation_mode: str
    use_refinement: bool
    refinement_techniques: List[str]
    use_personalization: bool
    personalization_user_id: Optional[str]
    personalization_alpha: float
    use_rag: bool
    rag_top_context_docs: int
    k1: float
    b: float
    top_n_filter: int
    display_top_k: int
    k_rrf: int
    bm25_rrf_weight: float
    embedding_rrf_weight: float


def _fetch_user_profile(profile_url: str) -> dict:
    try:
        response = requests.get(profile_url, timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


def _format_retrieval_option(mode_key: str) -> str:
    mode = RETRIEVAL_MODE_BY_KEY[mode_key]
    return f"{mode[1]} — {mode[2]}"


def render_sidebar(
    *,
    personalization_url: str,
    personalize_profile_url_fn,
    rag_ready: bool = True,
    rag_gemini_configured: bool = True,
) -> SearchSettings:
    if "ui_defaults_applied" not in st.session_state:
        st.session_state.ui_defaults_applied = True
        st.session_state.default_retrieval = DEFAULT_RETRIEVAL_MODE
        st.session_state.default_search_mode = DEFAULT_SEARCH_MODE
        st.session_state.default_refinement_preset = DEFAULT_REFINEMENT_PRESET
        st.session_state.default_personalization_on = True
        st.session_state.default_user_choice = DEFAULT_PERSONALIZATION_USER

    st.sidebar.header("إعدادات البحث")

    st.sidebar.markdown(RETRIEVAL_SECTION, unsafe_allow_html=True)
    default_retrieval_idx = RETRIEVAL_MODE_KEYS.index(
        st.session_state.get("default_retrieval", DEFAULT_RETRIEVAL_MODE)
    )
    representation_mode = st.sidebar.selectbox(
        "طريقة الاسترجاع",
        options=RETRIEVAL_MODE_KEYS,
        index=default_retrieval_idx,
        format_func=_format_retrieval_option,
        label_visibility="collapsed",
    )
    mode_meta = RETRIEVAL_MODE_BY_KEY[representation_mode]
    st.sidebar.caption(f"ماذا يفعل؟ {mode_meta[3]}")
    st.sidebar.caption(f"لماذا؟ {mode_meta[4]}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(REFINEMENT_SECTION, unsafe_allow_html=True)

    search_mode_options = list(SEARCH_MODES.keys())
    default_mode_idx = search_mode_options.index(
        st.session_state.get("default_search_mode", DEFAULT_SEARCH_MODE)
    )
    search_mode = st.sidebar.radio(
        "وضع البحث",
        search_mode_options,
        index=default_mode_idx,
        label_visibility="collapsed",
    )
    use_refinement = SEARCH_MODES[search_mode]
    mode_copy = SEARCH_MODE_COPY[search_mode]
    st.sidebar.caption(f"ماذا يفعل؟ {mode_copy['purpose']}")
    st.sidebar.caption(f"لماذا؟ {mode_copy['benefit']}")

    refinement_techniques: List[str] = []
    if use_refinement:
        preset_options = list(REFINEMENT_PRESETS.keys())
        default_preset_idx = preset_options.index(
            st.session_state.get("default_refinement_preset", DEFAULT_REFINEMENT_PRESET)
        )
        preset_name = st.sidebar.selectbox(
            "ملف التحسين",
            options=preset_options,
            index=default_preset_idx,
        )
        st.sidebar.caption(REFINEMENT_PRESET_HINTS.get(preset_name, ""))
        preset_techniques = REFINEMENT_PRESETS[preset_name]
        if preset_techniques is None:
            refinement_techniques = st.sidebar.multiselect(
                "تحسينات مخصصة",
                options=REFINEMENT_TECHNIQUE_OPTIONS,
                default=["query_preprocess"],
                format_func=lambda t: REFINEMENT_TECHNIQUE_LABELS.get(t, t),
            )
        else:
            refinement_techniques = preset_techniques

    st.sidebar.markdown("---")
    st.sidebar.markdown(PERSONALIZATION_SECTION, unsafe_allow_html=True)

    use_personalization = st.sidebar.checkbox(
        "تفعيل التخصيص (إعادة ترتيب حسب الملف الشخصي)",
        value=st.session_state.get("default_personalization_on", True),
        help="Personalized re-ranking based on stored interest profile.",
    )

    personalization_user_id: Optional[str] = None
    if use_personalization:
        user_options = list(PERSONALIZATION_USER_OPTIONS.keys())
        default_user_idx = user_options.index(
            st.session_state.get("default_user_choice", DEFAULT_PERSONALIZATION_USER)
        )
        user_choice = st.sidebar.selectbox(
            "المستخدم",
            options=user_options,
            index=default_user_idx,
        )
        selected = PERSONALIZATION_USER_OPTIONS[user_choice]
        if selected == "__custom__":
            personalization_user_id = st.sidebar.text_input(
                "معرف المستخدم",
                value="user_custom",
            ).strip() or None
        else:
            personalization_user_id = selected
            hint = PERSONALIZATION_USER_HINTS.get(selected)
            if hint:
                st.sidebar.caption(hint)

        if personalization_user_id and st.sidebar.button("إعادة تعيين الملف الشخصي"):
            try:
                requests.delete(
                    f"{personalization_url.rstrip('/')}/profile/{personalization_user_id}",
                    timeout=5,
                )
                st.sidebar.success("تمت إعادة التعيين — ابدأ العرض التوضيحي من جديد")
            except Exception as exc:
                st.sidebar.error(f"فشل إعادة التعيين: {exc}")

        if personalization_user_id:
            profile = _fetch_user_profile(
                personalize_profile_url_fn(personalization_user_id),
            )
            terms = profile.get("interest_terms") or {}
            if terms:
                top_terms = list(terms.keys())[:5]
                st.sidebar.caption(f"أهم مصطلحات الملف: {', '.join(top_terms)}")
            else:
                st.sidebar.caption("الملف فارغ — ابحث أو سجّل وثائق مفيدة")
            st.sidebar.caption(
                f"استعلامات: {profile.get('query_count', 0)} | "
                f"نقرات: {profile.get('click_count', 0)}"
            )

        personalization_alpha = st.sidebar.slider(
            "وزن التخصيص (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=float(ir_config.PERSONALIZATION_ALPHA),
            step=0.05,
            help="0 = ترتيب الاسترجاع فقط، 1 = أقصى تأثير لملف الاهتمام",
        )
    else:
        personalization_alpha = float(ir_config.PERSONALIZATION_ALPHA)
    st.sidebar.markdown("---")
    st.sidebar.markdown(RAG_SECTION, unsafe_allow_html=True)

    rag_disabled = not rag_ready or not rag_gemini_configured
    use_rag = st.sidebar.checkbox(
        "تفعيل الإجابة الذكية (RAG)",
        value=False,
        disabled=rag_disabled,
        help="Generates a natural-language answer from top retrieved passages via Gemini.",
    )
    if rag_disabled:
        if not rag_ready:
            st.sidebar.caption("خدمة RAG غير متصلة — شغّل rag_service على المنفذ 8006")
        elif not rag_gemini_configured:
            st.sidebar.caption("GEMINI_API_KEY غير مضبوط — أضفه في ملف .env")

    rag_top_context_docs = DEFAULT_RAG_CONTEXT_DOCS
    if use_rag:
        default_rag_idx = RAG_CONTEXT_DOC_OPTIONS.index(DEFAULT_RAG_CONTEXT_DOCS)
        rag_top_context_docs = st.sidebar.selectbox(
            "عدد المقاطع المستخدمة في الإجابة",
            options=RAG_CONTEXT_DOC_OPTIONS,
            index=default_rag_idx,
        )

    st.sidebar.markdown("---")
    k1 = 1.5
    b = 0.75
    top_n_filter = SERIAL_HYBRID_TOP_N
    display_top_k = DEFAULT_DISPLAY_TOP_K
    k_rrf = int(ir_config.RRF_K)
    bm25_rrf_weight = 1.0
    embedding_rrf_weight = 1.0

    with st.sidebar.expander("إعدادات متقدمة", expanded=False):
        st.caption("ماذا يفعل هذا؟ ضبط دقيق لخبراء IR والعرض الأكاديمي.")
        st.caption("لماذا؟ للتجارب المتقدمة — الافتراضيات مناسبة للعرض التوضيحي.")

        if representation_mode in ("bm25", "hybrid_parallel", "hybrid_serial"):
            k1 = st.slider(
                "BM25 k1 (term frequency scaling)",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1,
            )
            b = st.slider(
                "BM25 b (document length penalty)",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
            )

        if representation_mode == "hybrid_serial":
            top_n_filter = st.slider(
                "عدد مرشحي BM25 لإعادة الترتيب (serial hybrid)",
                min_value=10,
                max_value=500,
                value=SERIAL_HYBRID_TOP_N,
                step=10,
            )

        if representation_mode == "hybrid_parallel":
            k_rrf = st.slider(
                "ثابت دمج RRF (k_rrf)",
                min_value=10,
                max_value=100,
                value=int(ir_config.RRF_K),
                step=5,
                help="كلما زاد k_rrf قلّ تأثير الفرق بين الرتب العالية",
            )
            bm25_rrf_weight = st.slider(
                "وزن BM25 في الدمج الهجين",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
            )
            embedding_rrf_weight = st.slider(
                "وزن Embedding في الدمج الهجين",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
            )

        display_top_k = st.slider(
            "عدد النتائج المعروضة (top_k)",
            min_value=5,
            max_value=100,
            value=DEFAULT_DISPLAY_TOP_K,
            step=5,
        )

    return SearchSettings(
        representation_mode=representation_mode,
        use_refinement=use_refinement,
        refinement_techniques=refinement_techniques,
        use_personalization=use_personalization,
        personalization_user_id=personalization_user_id,
        personalization_alpha=personalization_alpha,
        use_rag=use_rag,
        rag_top_context_docs=rag_top_context_docs,
        k1=k1,
        b=b,
        top_n_filter=top_n_filter,
        display_top_k=display_top_k,
        k_rrf=k_rrf,
        bm25_rrf_weight=bm25_rrf_weight,
        embedding_rrf_weight=embedding_rrf_weight,
    )
