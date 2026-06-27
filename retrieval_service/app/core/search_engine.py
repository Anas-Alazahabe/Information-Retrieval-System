import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.index_store import IndexStore, JsonIndexStore
from shared.ir_config import (
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    FAISS_FILENAME,
    FAISS_ID_MAP_FILENAME,
    INDEX_DIR,
    SERIAL_HYBRID_TOP_N,
)


class BM25SearchEngine:
    """محرك استرجاع نصي يعتمد BM25."""

    def __init__(self, store: Optional[IndexStore] = None, index_dir: str = INDEX_DIR):
        """تهيئة المحرك وربطه بمخزن الفهارس."""
        self.store = store or JsonIndexStore(index_dir)
        self._load_from_store()

    def _load_from_store(self):
        """تحميل فهرس BM25 والميتا-بيانات في الذاكرة."""
        self.bm25_index = self.store.load_bm25()
        self.metadata = self.store.load_metadata()
        self.total_docs = self.metadata.get("total_docs", 0)
        self.avg_doc_len = self.metadata.get("avg_doc_len", 0)

    def reload(self):
        """إعادة تحميل الفهارس من التخزين."""
        self._load_from_store()

    def search(self, query_tokens: list, k1: float = 1.5, b: float = 0.75) -> dict:
        """حساب درجات BM25 وإرجاع النتائج مرتبة تنازليًا."""
        if not query_tokens or self.total_docs == 0:
            return {}

        doc_scores = {}

        for term in query_tokens:
            if term not in self.bm25_index:
                continue

            postings = self.bm25_index[term]
            df = len(postings)

            numerator_idf = self.total_docs - df + 0.5
            denominator_idf = df + 0.5
            bm25_idf = math.log((numerator_idf / denominator_idf) + 1.0)

            for doc_id, doc_info in postings.items():
                tf = doc_info["tf"]
                doc_len = doc_info["doc_len"]
                denominator = tf + k1 * (1.0 - b + (b * (doc_len / self.avg_doc_len)))
                term_score = bm25_idf * (tf * (k1 + 1.0)) / denominator
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + term_score

        sorted_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
        return {doc_id: round(score, 4) for doc_id, score in sorted_scores.items()}


class EmbeddingSearchEngine:
    """محرك استرجاع دلالي يعتمد Embedding Cosine Similarity."""

    def __init__(self, store: Optional[IndexStore] = None, index_dir: str = INDEX_DIR):
        """تهيئة المحرك مع تحميل متجهات الوثائق."""
        self.store = store or JsonIndexStore(index_dir)
        self.index_dir = store.index_dir if store else index_dir
        self.model = None
        self.faiss_index = None
        self.faiss_id_map: List[str] = []
        self._numpy_matrix = None
        self._numpy_doc_ids: List[str] = []
        self._load_from_store()

    def _load_from_store(self):
        """تحميل embeddings من التخزين (FAISS فقط عند الحجم الكبير)."""
        self.doc_embeddings = {}
        self.faiss_index = None
        self.faiss_id_map = []
        self._numpy_matrix = None
        self._numpy_doc_ids = []
        self._load_faiss_index()
        if self.faiss_index is not None and self.faiss_id_map:
            # At scale, vectors live in FAISS — avoid loading multi-GB JSON into RAM.
            pass
        else:
            self.doc_embeddings = self.store.load_embeddings()
        self._maybe_build_numpy_matrix()

    def has_embeddings(self) -> bool:
        return bool(self.doc_embeddings) or self.faiss_index is not None

    def reload(self):
        """إعادة تحميل embeddings من التخزين."""
        self._load_from_store()

    def _load_faiss_index(self):
        faiss_path = os.path.join(self.index_dir, FAISS_FILENAME)
        id_map_path = os.path.join(self.index_dir, FAISS_ID_MAP_FILENAME)
        if not os.path.exists(faiss_path) or not os.path.exists(id_map_path):
            return
        try:
            import faiss

            self.faiss_index = faiss.read_index(faiss_path)
            with open(id_map_path, "r", encoding="utf-8") as f:
                self.faiss_id_map = json.load(f)
        except Exception as exc:
            print(f"Warning: could not load FAISS index: {exc}")
            self.faiss_index = None
            self.faiss_id_map = []

    def _maybe_build_numpy_matrix(self):
        if EMBEDDING_BACKEND != "numpy" or not self.doc_embeddings:
            return
        try:
            import numpy as np

            doc_ids = list(self.doc_embeddings.keys())
            vectors = [self.doc_embeddings[d] for d in doc_ids]
            if not vectors:
                return
            matrix = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._numpy_matrix = matrix / norms
            self._numpy_doc_ids = doc_ids
        except Exception as exc:
            print(f"Warning: numpy matrix build failed: {exc}")
            self._numpy_matrix = None
            self._numpy_doc_ids = []

    def ann_backend(self) -> str:
        if self.faiss_index is not None:
            return "faiss"
        if self._numpy_matrix is not None:
            return "numpy"
        return "loop"

    def _lazy_load_model(self):
        """تحميل نموذج التضمين عند الطلب فقط (Lazy Loading)."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                self.model = None

    def _cosine_similarity(self, vec_a: list, vec_b: list) -> float:
        """حساب تشابه جيبي بين متجهين."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _score_loop(
        self, query_vector: list, doc_ids: Optional[list] = None
    ) -> Dict[str, float]:
        candidates = doc_ids if doc_ids is not None else self.doc_embeddings.keys()
        doc_scores = {}
        for doc_id in candidates:
            doc_vector = self.doc_embeddings.get(doc_id)
            if not doc_vector:
                continue
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                doc_scores[doc_id] = round(similarity, 4)
        return dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))

    def _score_numpy(
        self, query_vector: list, doc_ids: Optional[list] = None, top_k: Optional[int] = None
    ) -> Dict[str, float]:
        import numpy as np

        q = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return {}
        q = q / norm

        if doc_ids is not None:
            id_to_idx = {d: i for i, d in enumerate(self._numpy_doc_ids)}
            indices = [id_to_idx[d] for d in doc_ids if d in id_to_idx]
            if not indices:
                return {}
            sub_matrix = self._numpy_matrix[indices]
            sub_ids = [self._numpy_doc_ids[i] for i in indices]
            scores = sub_matrix @ q
            pairs = sorted(zip(sub_ids, scores), key=lambda x: -x[1])
            return {doc_id: round(float(score), 4) for doc_id, score in pairs if score > 0}

        scores = self._numpy_matrix @ q
        pairs = list(zip(self._numpy_doc_ids, scores))
        pairs.sort(key=lambda x: -x[1])
        if top_k:
            pairs = pairs[:top_k]
        return {doc_id: round(float(score), 4) for doc_id, score in pairs if score > 0}

    def _score_faiss(
        self, query_vector: list, top_k: Optional[int] = None
    ) -> Dict[str, float]:
        import numpy as np

        k = top_k or min(1000, len(self.faiss_id_map))
        if k <= 0:
            return {}
        q = np.array([query_vector], dtype=np.float32)
        distances, indices = self.faiss_index.search(q, k)
        doc_scores = {}
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.faiss_id_map):
                continue
            if score > 0:
                doc_scores[self.faiss_id_map[idx]] = round(float(score), 4)
        return dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))

    def _score_faiss_candidates(
        self, query_vector: list, doc_ids: list
    ) -> Dict[str, float]:
        import numpy as np

        id_to_idx = {doc_id: i for i, doc_id in enumerate(self.faiss_id_map)}
        q = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return {}
        q = q / norm

        doc_scores = {}
        for doc_id in doc_ids:
            idx = id_to_idx.get(doc_id)
            if idx is None:
                continue
            vec = self.faiss_index.reconstruct(int(idx))
            vec = np.array(vec, dtype=np.float32)
            vnorm = np.linalg.norm(vec)
            if vnorm == 0:
                continue
            score = float(np.dot(q, vec / vnorm))
            if score > 0:
                doc_scores[doc_id] = round(score, 4)
        return dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))

    def _score_vectors(
        self,
        query_vector: list,
        doc_ids: Optional[list] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        if doc_ids is not None:
            if self.faiss_index is not None and self.faiss_id_map:
                return self._score_faiss_candidates(query_vector, doc_ids)
            if self._numpy_matrix is not None and EMBEDDING_BACKEND == "numpy":
                return self._score_numpy(query_vector, doc_ids=doc_ids)
            return self._score_loop(query_vector, doc_ids=doc_ids)

        if self.faiss_index is not None and doc_ids is None:
            return self._score_faiss(query_vector, top_k=top_k)

        if self._numpy_matrix is not None and EMBEDDING_BACKEND == "numpy":
            return self._score_numpy(query_vector, top_k=top_k)

        return self._score_loop(query_vector, doc_ids=None)

    def search(self, query_text: str, doc_ids: Optional[list] = None) -> dict:
        """استرجاع دلالي على كل الوثائق أو على قائمة مرشحين فقط."""
        if not query_text.strip() or not self.has_embeddings():
            return {}

        self._lazy_load_model()
        if not self.model:
            return {}

        query_vector = self.model.encode(query_text, normalize_embeddings=True).tolist()
        return self._score_vectors(query_vector, doc_ids=doc_ids)


class HybridSearchEngine:
    """محرك هجيني يجمع بين BM25 وEmbedding بطريقتين."""

    def __init__(self, bm25_engine: BM25SearchEngine, embedding_engine: EmbeddingSearchEngine):
        """تهيئة المحرك الهجين بمحركيه الفرعيين."""
        self.bm25_engine = bm25_engine
        self.embedding_engine = embedding_engine

    def search_parallel(
        self,
        query_tokens: list,
        query_text: str,
        k1: float = 1.5,
        b: float = 0.75,
        k_rrf: int = 60,
    ) -> dict:
        """دمج متوازي باستخدام RRF على قوائم الترتيب من المحركين."""
        bm25_results = self.bm25_engine.search(query_tokens, k1=k1, b=b)
        embedding_results = self.embedding_engine.search(query_text)

        bm25_rank_list = list(bm25_results.keys())
        emb_rank_list = list(embedding_results.keys())
        bm25_ranks = {doc_id: i + 1 for i, doc_id in enumerate(bm25_rank_list)}
        emb_ranks = {doc_id: i + 1 for i, doc_id in enumerate(emb_rank_list)}

        all_docs = set(bm25_rank_list).union(set(emb_rank_list))
        rrf_scores = {}

        for doc_id in all_docs:
            rank_bm25 = bm25_ranks.get(doc_id)
            rank_emb = emb_ranks.get(doc_id)
            score_bm25 = 1.0 / (k_rrf + rank_bm25) if rank_bm25 else 0.0
            score_emb = 1.0 / (k_rrf + rank_emb) if rank_emb else 0.0
            rrf_scores[doc_id] = round(score_bm25 + score_emb, 6)

        return dict(sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True))

    def search_serial(
        self,
        query_tokens: list,
        query_text: str,
        k1: float = 1.5,
        b: float = 0.75,
        top_n_filter: int = SERIAL_HYBRID_TOP_N,
    ) -> dict:
        """استرجاع تسلسلي: BM25 للترشيح ثم إعادة ترتيب دلالية."""
        bm25_results = self.bm25_engine.search(query_tokens, k1=k1, b=b)
        top_docs_from_bm25 = list(bm25_results.keys())[:top_n_filter]

        if not top_docs_from_bm25:
            return {}

        self.embedding_engine._lazy_load_model()
        if not self.embedding_engine.model:
            return bm25_results

        return self.embedding_engine.search(query_text, doc_ids=top_docs_from_bm25)
