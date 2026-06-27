"""Cached evaluation insights for the Streamlit UI."""

import json
import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from shared.ir_config import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "evaluation_results"
CHARTS_DIR = EVAL_DIR / "charts"

MODE_LABELS = {
    "vsm": "VSM (TF-IDF)",
    "bm25": "BM25",
    "embedding": "Embedding",
    "hybrid_parallel": "Hybrid (RRF)",
    "hybrid_serial": "Hybrid (Serial)",
}


def _latest_json(pattern: str) -> Optional[Dict[str, Any]]:
    paths = sorted(glob(str(EVAL_DIR / pattern)), reverse=True)
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            continue
    return None


def _best_mode(baseline: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
    modes = baseline.get("modes") or {}
    if not modes:
        return "embedding", {}
    best = max(
        modes.items(),
        key=lambda item: item[1].get("ndcg_at_10", 0),
    )
    return best[0], best[1]


def render_eval_insights() -> None:
    st.markdown("### تحليل الأداء (من آخر تقييم)")
    st.caption(
        "أرقام من `evaluation_results/` — مجموعة dev qrels الرسمية، "
        "وليس استعلامات الواجهة."
    )

    baseline = _latest_json("eval_baseline_full_*.json")
    vector_store = _latest_json("vector_store_ablation_full_*.json")
    refinement = _latest_json("eval_refined_*_full_*.json")
    personalization = _latest_json("eval_personalization_full_*.json")

    if not baseline and not vector_store:
        st.info(
            "لا توجد نتائج تقييم محفوظة. نفّذ: "
            "`python scripts/run_full_evaluation.py`"
        )
        return

    if baseline:
        best_mode, best_metrics = _best_mode(baseline)
        st.markdown(
            f"**أفضل تمثيل (nDCG@10):** {MODE_LABELS.get(best_mode, best_mode)} "
            f"= {best_metrics.get('ndcg_at_10', 0):.4f}"
        )
        with st.expander("لماذا FAISS / Embedding؟", expanded=False):
            emb = (baseline.get("modes") or {}).get("embedding", {})
            bm = (baseline.get("modes") or {}).get("bm25", {})
            delta = emb.get("ndcg_at_10", 0) - bm.get("ndcg_at_10", 0)
            st.write(
                f"Embedding يتفوق على BM25 بمقدار ΔnDCG@10 = {delta:+.4f} "
                "لأن FAISS يسترجع دلالياً عندما تختلف صياغة السؤال عن النص."
            )
            if CHARTS_DIR.joinpath("04_vector_store_comparison.png").exists():
                st.image(
                    str(CHARTS_DIR / "04_vector_store_comparison.png"),
                    caption="مقارنة التمثيلات",
                )

        with st.expander("لماذا الهجين (Hybrid)؟", expanded=False):
            hp = (baseline.get("modes") or {}).get("hybrid_parallel", {})
            st.write(
                "الهجين المتموازي (RRF) يدمج ترتيب BM25 و Embedding: "
                f"nDCG@10 = {hp.get('ndcg_at_10', 0):.4f}. "
                "مفيد عندما تريد دقة كلمات + معنى معاً."
            )

    if vector_store:
        faiss = vector_store.get("faiss_status") or {}
        st.caption(
            f"FAISS: backend={faiss.get('ann_backend')} · "
            f"loaded={faiss.get('faiss_loaded')}"
        )

    if refinement:
        deltas = refinement.get("deltas_vs_baseline", {}).get("combined", {})
        if deltas:
            sample = next(iter(deltas.values()), {})
            st.markdown(
                f"**تحسين الاستعلام (combined):** ΔnDCG@10 = "
                f"{sample.get('ndcg_at_10', 0):+.4f}"
            )

    if personalization:
        pers = personalization.get("deltas_personalized_vs_baseline") or {}
        if pers:
            sample = next(iter(pers.values()), {})
            st.markdown(
                f"**التخصيص:** ΔnDCG@10 = {sample.get('ndcg_at_10', 0):+.4f} "
                "(مستخدمون محاكون)"
            )

    summary_path = EVAL_DIR / "FINAL_EVAL_SUMMARY.md"
    if summary_path.exists():
        with st.expander("ملخص التقييم الكامل"):
            st.markdown(summary_path.read_text(encoding="utf-8"))
