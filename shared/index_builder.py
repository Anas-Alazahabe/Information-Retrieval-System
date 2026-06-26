import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from shared.ir_config import (
    ARTIFACT_FILES,
    DATASET_NAME,
    EMBEDDING_MODEL,
    FAISS_FILENAME,
    FAISS_ID_MAP_FILENAME,
    FAISS_THRESHOLD,
    INDEX_DIR,
    PREPROCESS_FLAGS,
    PREPROCESS_URL,
    get_git_commit,
    get_max_docs_for_scale,
    preprocess_batch_url,
)


class IndexBuilder:
    """يبني جميع تمثيلات الفهرسة في الذاكرة ثم يحفظها على القرص.

    يدعم:
    - الفهرس المعكوس الخام
    - VSM (TF-IDF)
    - BM25 بصيغة مناسبة للحساب وقت الاستعلام
    - بيانات وصفية (metadata) وملف manifest
    """

    def __init__(self):
        """تهيئة هياكل البيانات الداخلية لعملية الفهرسة."""
        self.raw_inverted_index: Dict[str, Dict[str, int]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.doc_embeddings: Dict[str, List[float]] = {}
        self.empty_token_doc_ids: set = set()
        self.total_docs = 0
        self.total_length = 0

    def add_documents(
        self,
        doc_ids: List[str],
        tokens_batch: List[List[str]],
        embeddings_batch: Optional[List[List[float]]] = None,
    ) -> int:
        """يضيف دفعة وثائق إلى الهياكل الداخلية.

        المدخلات:
        - `doc_ids`: معرفات الوثائق
        - `tokens_batch`: التوكنز المعالجة لكل وثيقة
        - `embeddings_batch`: المتجهات الدلالية المقابلة (اختياري)
        """
        if embeddings_batch is None:
            embeddings_batch = [[] for _ in doc_ids]

        indexed_count = 0
        for doc_id, tokens, embedding in zip(doc_ids, tokens_batch, embeddings_batch):
            if not tokens:
                tokens = ["empty_doc"]
                self.empty_token_doc_ids.add(doc_id)

            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            self.total_length += doc_len
            self.total_docs += 1

            if embedding:
                self.doc_embeddings[doc_id] = embedding

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term not in self.raw_inverted_index:
                    self.raw_inverted_index[term] = {}
                self.raw_inverted_index[term][doc_id] = count

            indexed_count += 1

        return indexed_count

    def export_state(self) -> Dict[str, Any]:
        """Serialize in-memory index state for checkpoint/resume."""
        return {
            "raw_inverted_index": self.raw_inverted_index,
            "doc_lengths": self.doc_lengths,
            "doc_embeddings": self.doc_embeddings,
            "empty_token_doc_ids": list(self.empty_token_doc_ids),
            "total_docs": self.total_docs,
            "total_length": self.total_length,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore in-memory index state from a checkpoint."""
        self.raw_inverted_index = state["raw_inverted_index"]
        self.doc_lengths = state["doc_lengths"]
        self.doc_embeddings = state["doc_embeddings"]
        self.empty_token_doc_ids = set(state.get("empty_token_doc_ids", []))
        self.total_docs = int(state["total_docs"])
        self.total_length = int(state["total_length"])

    def _compute_indices(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """يحسب تمثيلات VSM/BM25 والميتا-بيانات وNorms للوثائق."""
        final_vsm_index: Dict[str, Dict[str, float]] = {}
        final_bm25_index: Dict[str, Dict[str, Dict[str, Any]]] = {}
        idf_weights: Dict[str, float] = {}
        doc_weight_squares: Dict[str, float] = {}

        for term, postings in self.raw_inverted_index.items():
            df = len(postings)
            idf = math.log10(self.total_docs / df) if df > 0 else 0
            idf_weights[term] = round(idf, 4)
            final_vsm_index[term] = {}
            final_bm25_index[term] = {}

            for doc_id, tf in postings.items():
                log_tf = 1 + math.log10(tf) if tf > 0 else 0
                weight = round(log_tf * idf, 4)
                final_vsm_index[term][doc_id] = weight
                doc_weight_squares[doc_id] = doc_weight_squares.get(doc_id, 0.0) + weight ** 2

                final_bm25_index[term][doc_id] = {
                    "tf": tf,
                    "doc_len": self.doc_lengths[doc_id],
                }

        doc_norms = {
            doc_id: round(math.sqrt(square_sum), 6) if square_sum > 0 else 0.0
            for doc_id, square_sum in doc_weight_squares.items()
        }

        avg_doc_len = self.total_length / self.total_docs if self.total_docs > 0 else 0
        metadata = {
            "total_docs": self.total_docs,
            "avg_doc_len": round(avg_doc_len, 2),
            "doc_lengths": self.doc_lengths,
            "idf_weights": idf_weights,
            "doc_norms": doc_norms,
        }

        return final_vsm_index, final_bm25_index, metadata, doc_norms

    def validate(self) -> Dict[str, Any]:
        """ينفذ فحوصات سلامة بعد الفهرسة ويعيد إحصاءات تشخيصية."""
        missing_embeddings = [
            doc_id
            for doc_id in self.doc_lengths
            if doc_id not in self.doc_embeddings or not self.doc_embeddings[doc_id]
        ]
        stats = {
            "indexed_docs_count": self.total_docs,
            "empty_token_docs_count": len(self.empty_token_doc_ids),
            "missing_embeddings_count": len(missing_embeddings),
            "unique_terms_count": len(self.raw_inverted_index),
        }
        if self.total_docs == 0:
            raise ValueError("Index is empty: indexed_docs_count is 0")
        if missing_embeddings:
            print(
                f"Warning: {len(missing_embeddings)} documents are missing embeddings."
            )
        return stats

    def _build_faiss_index(self, index_dir: str) -> Dict[str, Any]:
        """Build FAISS inner-product index from document embeddings."""
        if not self.doc_embeddings or len(self.doc_embeddings) < FAISS_THRESHOLD:
            return {"ann_backend": "none", "vector_count": len(self.doc_embeddings)}

        try:
            import faiss
            import numpy as np

            doc_ids = list(self.doc_embeddings.keys())
            vectors = np.array(
                [self.doc_embeddings[d] for d in doc_ids], dtype=np.float32
            )
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)

            faiss.write_index(index, os.path.join(index_dir, FAISS_FILENAME))
            with open(
                os.path.join(index_dir, FAISS_ID_MAP_FILENAME), "w", encoding="utf-8"
            ) as f:
                json.dump(doc_ids, f)

            return {
                "ann_backend": "faiss",
                "embedding_dim": dim,
                "vector_count": len(doc_ids),
            }
        except Exception as exc:
            print(f"Warning: FAISS index build skipped: {exc}")
            return {
                "ann_backend": "json_fallback",
                "vector_count": len(self.doc_embeddings),
            }

    def _fetch_preprocessing_health(self) -> Dict[str, Any]:
        """يجلب حالة خدمة المعالجة المسبقة لإدراجها في manifest."""
        try:
            response = requests.get(f"{PREPROCESS_URL.rstrip('/')}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}

    def save(
        self,
        index_dir: str = INDEX_DIR,
        dataset_name: str = DATASET_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        index_scale_mode: str = "dev",
        max_docs_cap: Optional[int] = None,
        preprocessing_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """يحفظ كل ملفات الفهرسة ويولّد ملف `index_manifest.json`."""
        if self.total_docs == 0:
            raise ValueError("Cannot save an empty index.")

        os.makedirs(index_dir, exist_ok=True)
        sanity = self.validate()

        final_vsm_index, final_bm25_index, metadata, _ = self._compute_indices()

        with open(os.path.join(index_dir, "vsm_index.json"), "w", encoding="utf-8") as f:
            json.dump(final_vsm_index, f, ensure_ascii=False)

        with open(os.path.join(index_dir, "bm25_index.json"), "w", encoding="utf-8") as f:
            json.dump(final_bm25_index, f, ensure_ascii=False)

        with open(os.path.join(index_dir, "embeddings_index.json"), "w", encoding="utf-8") as f:
            json.dump(self.doc_embeddings, f)

        with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

        ann_info = self._build_faiss_index(index_dir)

        health = self._fetch_preprocessing_health()
        preprocessing = preprocessing_override or {
            **PREPROCESS_FLAGS,
            "spacy_available": health.get("spacy_available", False),
            "lemmatization_mode": health.get("lemmatization_mode", "unknown"),
        }

        manifest = {
            "dataset_name": dataset_name,
            "document_count": self.total_docs,
            "preprocessing": preprocessing,
            "embedding_model": embedding_model,
            "index_scale_mode": index_scale_mode,
            "max_docs_cap": max_docs_cap,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "artifact_files": list(ARTIFACT_FILES),
            "sanity_checks": sanity,
            "matcher_version": "2.1",
            **ann_info,
        }

        with open(os.path.join(index_dir, "index_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"Index saved to {index_dir} ({self.total_docs} documents)")
        return manifest


def fetch_preprocessing_mode() -> str:
    """مساعد سريع لقراءة نمط lemmatization الحالي من خدمة المعالجة."""
    try:
        response = requests.get(f"{PREPROCESS_URL.rstrip('/')}/health", timeout=5)
        response.raise_for_status()
        return response.json().get("lemmatization_mode", "unknown")
    except Exception:
        return "unknown"
