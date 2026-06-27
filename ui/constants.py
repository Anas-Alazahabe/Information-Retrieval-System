"""Arabic-first labels, presets, and section copy for the search UI."""

from typing import Any, Dict, List, Optional, Tuple

import shared.ir_config as ir_config

VALID_REFINEMENT_TECHNIQUES = ir_config.VALID_REFINEMENT_TECHNIQUES
SERIAL_HYBRID_TOP_N = ir_config.SERIAL_HYBRID_TOP_N
SUGGEST_DEFAULT_LIMIT = ir_config.SUGGEST_DEFAULT_LIMIT
SUGGEST_MIN_PREFIX_LEN = ir_config.SUGGEST_MIN_PREFIX_LEN

STATUS_CACHE_KEY = "system_status_cache"
STATUS_CACHE_TTL_SEC = 45
SUGGESTION_DISPLAY_MAX_LEN = 60
ENHANCED_QUERY_SUMMARY_MAX_LEN = 80
SNIPPET_MAX_LEN = 380

CLICK_BUTTON_LABEL = "هذا مفيد — حدّث اهتماماتي"
CLICK_BUTTON_HELP = (
    "سجّل أن هذه الوثيقة مفيدة لتعليم النظام اهتماماتك في عمليات البحث القادمة (وزن ×2). "
    "Logs a click event (2× weight) to update your interest profile for future searches."
)

REFINEMENT_TECHNIQUE_LABELS = {
    "query_preprocess": "معالجة الأسئلة الطبيعية",
    "prf": "توسيع بالملاحظات",
    "synonyms": "توسيع بالمرادفات",
    "history": "سياق البحث السابق",
}

REFINEMENT_PRESETS: Dict[str, Optional[List[str]]] = {
    "موصى به": ["query_preprocess", "prf", "synonyms"],
    "أسئلة طبيعية": ["query_preprocess"],
    "سجل الجلسة": ["query_preprocess", "history"],
    "تخصيص متقدم": None,
}

REFINEMENT_PRESET_HINTS = {
    "موصى به": "أفضل توازن للعرض الأكاديمي — preprocess + PRF + synonyms",
    "أسئلة طبيعية": "للأسئلة بصيغة who / what / how",
    "سجل الجلسة": "يذكر عمليات البحث السابقة في هذه الجلسة",
    "تخصيص متقدم": "للمحاضر — اختيار التقنيات يدوياً",
}

SEARCH_MODES = {
    "بحث مباشر": False,
    "بحث محسّن": True,
}

SEARCH_MODE_COPY = {
    "بحث مباشر": {
        "purpose": "يبحث بنصك كما هو دون تعديل.",
        "benefit": "أسرع؛ مفيد للمقارنة قبل/بعد التحسين.",
    },
    "بحث محسّن": {
        "purpose": "يحسّن الاستعلام قبل الاسترجاع.",
        "benefit": "نتائج أفضل للأسئلة الطبيعية والمرادفات.",
    },
}

RETRIEVAL_MODES: List[Tuple[str, str, str, str, str]] = [
    (
        "bm25",
        "مطابقة كلمات (BM25)",
        "Okapi BM25",
        "يطابق الكلمات في النص حرفياً.",
        "أفضل للأسئلة المباشرة والمصطلحات الدقيقة.",
    ),
    (
        "embedding",
        "مطابقة دلالية",
        "Semantic Embedding",
        "يفهم المعنى وليس الحروف فقط.",
        "أفضل عندما تختلف صياغتك عن النص.",
    ),
    (
        "hybrid_parallel",
        "هجين متوازي (RRF)",
        "Hybrid Parallel",
        "يجمع الكلمات والمعنى معاً.",
        "توازن عام — موصى به للتجربة.",
    ),
    (
        "vsm",
        "نموذج الفضاء المتجه (TF-IDF)",
        "Vector Space Model",
        "مطابقة تقليدية بالتكرار.",
        "للمقارنة الأكاديمية.",
    ),
    (
        "hybrid_serial",
        "هجين تسلسلي",
        "Hybrid Serial + re-rank",
        "يصفّي بـ BM25 ثم يعيد الترتيب دلالياً.",
        "دقة أعلى لكن أبطأ.",
    ),
]

RETRIEVAL_MODE_KEYS = [mode[0] for mode in RETRIEVAL_MODES]
RETRIEVAL_MODE_BY_KEY = {mode[0]: mode for mode in RETRIEVAL_MODES}

REFINEMENT_TECHNIQUE_OPTIONS = list(VALID_REFINEMENT_TECHNIQUES)

PERSONALIZATION_USER_OPTIONS = {
    "مستخدم صحّة (demo_health)": "demo_health",
    "مستخدم تقنية (demo_tech)": "demo_tech",
    "مستخدم مخصص": "__custom__",
}

PERSONALIZATION_USER_HINTS = {
    "demo_health": "مهتم بالصحة — يفضّل وثائق طبية في العرض التوضيحي.",
    "demo_tech": "مهتم بالتقنية — يفضّل وثائق برمجة وتقنية.",
}

DEFAULT_RETRIEVAL_MODE = "bm25"
DEFAULT_SEARCH_MODE = "بحث محسّن"
DEFAULT_REFINEMENT_PRESET = "موصى به"
DEFAULT_PERSONALIZATION_USER = "مستخدم صحّة (demo_health)"


def retrieval_mode_label(mode_key: str) -> str:
    mode = RETRIEVAL_MODE_BY_KEY.get(mode_key)
    if not mode:
        return mode_key
    return f"{mode[1]} — {mode[2]}"


def section_header(title: str, purpose: str, benefit: str) -> str:
    return (
        f"**{title}**\n\n"
        f"<span class='section-purpose'>ماذا يفعل هذا؟ {purpose}</span><br>"
        f"<span class='section-benefit'>لماذا تستخدمه؟ {benefit}</span>"
    )


PERSONALIZATION_SECTION = section_header(
    "التخصيص",
    "يعيد ترتيب النتائج حسب ملف اهتماماتك المخزّن.",
    "نفس الاستعلام قد يعطي ترتيباً مختلفاً لكل مستخدم.",
)

RAG_SECTION = section_header(
    "الإجابة الذكية (RAG)",
    "يولّد إجابة بلغة طبيعية من المقاطع المسترجعة.",
    "مفيد عندما تريد ملخصاً مباشراً بدلاً من قراءة كل وثيقة.",
)

RAG_CONTEXT_DOC_OPTIONS = [3, 5, 8]
DEFAULT_RAG_CONTEXT_DOCS = 5

RETRIEVAL_SECTION = section_header(
    "كيف نجد النتائج؟",
    "يحدد طريقة مطابقة استعلامك مع النصوص.",
    "اختر الأنسب لنوع سؤالك — كلمات دقيقة أو معنى عام.",
)

REFINEMENT_SECTION = section_header(
    "تحسين الاستعلام",
    "يحسّن صياغة البحث قبل الاسترجاع.",
    "يساعدك على إيجاد وثائق لم تكن ستظهر بكلماتك الأصلية فقط.",
)
