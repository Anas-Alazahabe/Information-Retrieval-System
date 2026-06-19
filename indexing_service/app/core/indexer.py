import argparse
import sys
from pathlib import Path

import ir_datasets
import requests

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.index_builder import IndexBuilder
from shared.ir_config import (
    DATASET_NAME,
    EMBEDDING_MODEL,
    INDEX_DIR,
    PREPROCESS_FLAGS,
    get_max_docs_for_scale,
    preprocess_batch_url,
)


class DatasetIndexer:
    """منسّق الفهرسة من `ir_datasets` حتى تخزين المخرجات النهائية.

    هذا الكلاس:
    1) يقرأ الوثائق من الداتا سِت
    2) يولّد embeddings (إن توفرت المكتبة)
    3) يرسل النصوص لخدمة preprocessing
    4) يمرر النتائج إلى `IndexBuilder`
    """

    def __init__(self, dataset_name: str = DATASET_NAME):
        """تهيئة الداتا سِت وباني الفهرس."""
        self.dataset_name = dataset_name
        self.dataset = ir_datasets.load(dataset_name)
        self.builder = IndexBuilder()

    def process_and_index(
        self,
        batch_size: int = 500,
        max_docs: int | None = None,
        index_scale_mode: str = "dev",
        index_dir: str = INDEX_DIR,
    ):
        """ينفذ دورة الفهرسة كاملة وفق إعدادات الحجم المختارة."""
        if max_docs is None:
            max_docs = get_max_docs_for_scale(index_scale_mode)
        if max_docs is None:
            max_docs = float("inf")

        try:
            from sentence_transformers import SentenceTransformer

            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"Loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            embedding_model = None

        current_texts = []
        current_ids = []
        current_embeddings = []
        docs = self.dataset.docs_iter()
        loaded_docs = 0

        print(f"Starting indexing for {self.dataset_name} (scale={index_scale_mode}, max_docs={max_docs})")

        while loaded_docs < max_docs:
            try:
                doc = next(docs)
            except StopIteration:
                break
            except Exception:
                continue

            doc_text = getattr(doc, "text", getattr(doc, "body", ""))
            doc_text = str(doc_text).strip()

            vector = []
            if embedding_model:
                try:
                    vector = embedding_model.encode(doc_text, normalize_embeddings=True).tolist()
                except Exception:
                    vector = []

            current_texts.append(doc_text)
            current_ids.append(doc.doc_id)
            current_embeddings.append(vector)
            loaded_docs += 1

            if len(current_texts) >= batch_size:
                self._index_batch(current_ids, current_texts, current_embeddings)
                current_texts = []
                current_ids = []
                current_embeddings = []

        if current_texts:
            self._index_batch(current_ids, current_texts, current_embeddings)

        manifest = self.builder.save(
            index_dir=index_dir,
            dataset_name=self.dataset_name,
            embedding_model=EMBEDDING_MODEL,
            index_scale_mode=index_scale_mode,
            max_docs_cap=max_docs if max_docs != float("inf") else None,
        )
        print(f"Manifest written: {manifest.get('timestamp')}")

    def _index_batch(self, ids, texts, embeddings):
        """يعالج دفعة نصوص عبر preprocessing ثم يضيفها للفهرس."""
        payload = {
            "texts": texts,
            **PREPROCESS_FLAGS,
        }

        try:
            response = requests.post(preprocess_batch_url(), json=payload)
            response.raise_for_status()
            results = response.json()["results"]
            count = self.builder.add_documents(ids, results, embeddings)
            print(f"Indexed batch of {count} documents.")
        except Exception as e:
            print(f"Batch preprocessing failed: {e}")


def main():
    """نقطة الدخول لسطر الأوامر لبناء الفهارس."""
    parser = argparse.ArgumentParser(description="Build IR index from ir_datasets collection")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--scale", default="dev", choices=["dev", "preval", "full"])
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--index-dir", default=INDEX_DIR)
    args = parser.parse_args()

    indexer = DatasetIndexer(args.dataset)
    indexer.process_and_index(
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        index_scale_mode=args.scale,
        index_dir=args.index_dir,
    )


if __name__ == "__main__":
    main()
