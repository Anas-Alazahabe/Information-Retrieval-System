"""Read-only dataset scope panel for honest demo labeling."""

import json
import os
from typing import Any, Dict, Optional

import streamlit as st

from shared.ir_config import DATASET_NAME, EVAL_DATASET_NAME, INDEX_DIR, PROJECT_ROOT


def _load_manifest() -> Dict[str, Any]:
    path = os.path.join(INDEX_DIR, "index_manifest.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def render_dataset_info_panel() -> None:
    manifest = _load_manifest()
    doc_count = manifest.get("document_count", "—")
    scale_mode = manifest.get("index_scale_mode", "—")
    max_cap = manifest.get("max_docs_cap", "—")
    dataset = manifest.get("dataset_name", DATASET_NAME)

    with st.expander("معلومات مجموعة البيانات (للعرض والتقييم)", expanded=False):
        st.markdown(
            f"""
**المصدر:** [{dataset}](https://ir-datasets.com/msmarco-passage/) عبر `ir_datasets`

**ما هو مفهرس فعلياً:** أول **{doc_count:,}** مقطعاً (وضع `{scale_mode}`، حد أقصى {max_cap:,})

**ما ليس مفهرساً:** المجموعة الكاملة (~8.8M مقطع) — لم تُشغَّل على هذا العرض التوضيحي

**تقييم الأداء (IR metrics):** `{EVAL_DATASET_NAME}` — استعلامات و qrels رسمية فقط (لا استعلامات الواجهة)

**استثناء المشرف:** مجموعة بيانات واحدة (MS MARCO) بدلاً من اثنتين — موثّق في التقرير

**لماذا 200K؟** توازن بين حجم الفهرس، الذاكرة، ووقت العرض أمام اللجنة
            """
        )
        st.caption(f"مسار الفهرس: `{INDEX_DIR}` · جذر المشروع: `{PROJECT_ROOT}`")
