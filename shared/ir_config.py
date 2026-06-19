"""إعدادات مركزية مشتركة بين جميع خدمات مشروع IR.

هذا الملف هو المصدر الواحد للحقيقة (Single Source of Truth)
للمسارات، المنافذ، أنماط التشغيل، وأسماء النماذج.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_DIR = os.environ.get("IR_INDEX_DIR", str(PROJECT_ROOT / "index_data"))
PREPROCESS_URL = os.environ.get("IR_PREPROCESS_URL", "http://127.0.0.1:8000")
RETRIEVAL_URL = os.environ.get("IR_RETRIEVAL_URL", "http://127.0.0.1:8002")
REFINEMENT_URL = os.environ.get("IR_REFINEMENT_URL", "http://127.0.0.1:8003")
DATASET_NAME = os.environ.get("IR_DATASET", "msmarco-passage")
EVAL_DATASET_NAME = os.environ.get("IR_EVAL_DATASET", "msmarco-passage/dev")
EMBEDDING_MODEL = os.environ.get("IR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SERIAL_HYBRID_TOP_N = int(os.environ.get("IR_SERIAL_TOP_N", "100"))
RRF_K = int(os.environ.get("IR_RRF_K", "60"))
EMBEDDING_BACKEND = os.environ.get("IR_EMBEDDING_BACKEND", "numpy")
EMBEDDING_SEARCH_K = int(os.environ.get("IR_EMBEDDING_SEARCH_K", "1000"))
FAISS_FILENAME = "embeddings.faiss"
FAISS_ID_MAP_FILENAME = "embeddings_id_map.json"
FAISS_THRESHOLD = int(os.environ.get("IR_FAISS_THRESHOLD", "10000"))

MATCHER_METADATA: Dict[str, Dict[str, Any]] = {
    "vsm": {
        "matching_method": "cosine_similarity",
        "query_inputs": ["tokens"],
        "index_artifacts": ["vsm_index.json", "metadata.json"],
    },
    "bm25": {
        "matching_method": "bm25",
        "query_inputs": ["tokens"],
        "index_artifacts": ["bm25_index.json", "metadata.json"],
    },
    "embedding": {
        "matching_method": "cosine_similarity",
        "query_inputs": ["raw_text"],
        "index_artifacts": ["embeddings_index.json", "embeddings.faiss"],
    },
    "hybrid_parallel": {
        "matching_method": "rrf",
        "query_inputs": ["tokens", "raw_text"],
        "index_artifacts": ["bm25_index.json", "embeddings_index.json"],
    },
    "hybrid_serial": {
        "matching_method": "bm25_filter_cosine_rerank",
        "query_inputs": ["tokens", "raw_text"],
        "index_artifacts": ["bm25_index.json", "embeddings_index.json"],
    },
}

PREPROCESS_FLAGS: Dict[str, bool] = {
    "use_stemming": False,
    "use_lemmatization": True,
    "remove_stopwords": True,
}

INDEX_SCALE_CAPS = {
    "dev": 5_000,
    "preval": 30_000,
    "full": None,
}

INDEX_SCALE_MODE = os.environ.get("IR_INDEX_SCALE", "dev")

VALID_REPRESENTATION_MODES = (
    "vsm",
    "bm25",
    "embedding",
    "hybrid_parallel",
    "hybrid_serial",
)

VALID_REFINEMENT_TECHNIQUES = ("prf", "synonyms", "history", "query_preprocess")
DEFAULT_REFINEMENT_TECHNIQUES: tuple = ()

WH_WORDS = frozenset({"what", "how", "why", "when", "where", "who", "which"})

PRF_TOP_K_DOCS = int(os.environ.get("IR_PRF_TOP_K_DOCS", "10"))
PRF_TOP_M_TERMS = int(os.environ.get("IR_PRF_TOP_M_TERMS", "15"))
PRF_ORIGINAL_QUERY_WEIGHT = float(os.environ.get("IR_PRF_ORIGINAL_QUERY_WEIGHT", "0.5"))

SYNONYM_MAX_PER_TERM = int(os.environ.get("IR_SYNONYM_MAX_PER_TERM", "2"))
SYNONYM_MAX_TOTAL = int(os.environ.get("IR_SYNONYM_MAX_TOTAL", "8"))

QUERY_SUGGESTIONS_PATH = os.environ.get(
    "IR_QUERY_SUGGESTIONS",
    str(PROJECT_ROOT / "index_data" / "query_suggestions.json"),
)
SUGGEST_DEFAULT_LIMIT = int(os.environ.get("IR_SUGGEST_DEFAULT_LIMIT", "5"))
SUGGEST_MIN_PREFIX_LEN = int(os.environ.get("IR_SUGGEST_MIN_PREFIX_LEN", "2"))

HISTORY_MAX_QUERIES = int(os.environ.get("IR_HISTORY_MAX_QUERIES", "5"))
HISTORY_MAX_TERMS = int(os.environ.get("IR_HISTORY_MAX_TERMS", "5"))

ARTIFACT_FILES = (
    "vsm_index.json",
    "bm25_index.json",
    "embeddings_index.json",
    "metadata.json",
    "index_manifest.json",
    FAISS_FILENAME,
    FAISS_ID_MAP_FILENAME,
)


def get_max_docs_for_scale(scale: Optional[str] = None) -> Optional[int]:
    """يعيد حد عدد الوثائق المناسب لنمط الفهرسة.

    إذا تم تعريف `IR_MAX_DOCS` في البيئة يتم اعتماده مباشرة.
    """
    scale = (scale or INDEX_SCALE_MODE).lower()
    env_override = os.environ.get("IR_MAX_DOCS")
    if env_override:
        return int(env_override)
    return INDEX_SCALE_CAPS.get(scale, INDEX_SCALE_CAPS["dev"])


def get_git_commit() -> Optional[str]:
    """يحاول قراءة commit hash الحالي من Git.

    يعيد `None` إذا لم يكن المشروع Git repo أو عند فشل الأمر.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def preprocess_batch_url() -> str:
    """يبني رابط endpoint الخاص بالمعالجة الدُفعية للنصوص."""
    return f"{PREPROCESS_URL.rstrip('/')}/preprocess-batch"


def preprocess_single_url() -> str:
    """يبني رابط endpoint الخاص بمعالجة نص واحد."""
    return f"{PREPROCESS_URL.rstrip('/')}/preprocess"


def refine_url() -> str:
    """يبني رابط endpoint الخاص بتحسين الاستعلام."""
    return f"{REFINEMENT_URL.rstrip('/')}/refine"


def suggest_url() -> str:
    """يبني رابط endpoint الخاص باقتراحات الاستعلام."""
    return f"{REFINEMENT_URL.rstrip('/')}/suggest"
