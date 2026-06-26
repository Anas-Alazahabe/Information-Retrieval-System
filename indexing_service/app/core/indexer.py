import argparse
import sys
import requests
import ir_datasets
from pathlib import Path
from sentence_transformers import SentenceTransformer
import io
import codecs

# إجبار بايثون على استخدام utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# إعداد المسارات
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
    def __init__(self, dataset_name: str = DATASET_NAME):
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
        if max_docs is None:
            max_docs = get_max_docs_for_scale(index_scale_mode)
        if max_docs is None:
            max_docs = float("inf")

        # التعديل هنا: تحميل الموديل بشكل صريح وبدون إخفاء الأخطاء (Fail Fast)
        print(f"Loading embedding model from: {EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Loaded embedding model successfully!")

        docs = self.dataset.docs_iter()
        loaded_docs = 0

        print(f"Starting indexing: Preprocessing -> Embedding -> Indexing")

        while loaded_docs < max_docs:
            batch_raw_texts = []
            batch_ids = []

            # 1. تجميع النصوص في دفعة
            while len(batch_raw_texts) < batch_size and loaded_docs < max_docs:
                try:
                    doc = next(docs)
                    doc_text = getattr(doc, "text", getattr(doc, "body", ""))
                    batch_raw_texts.append(str(doc_text).strip())
                    batch_ids.append(doc.doc_id)
                    loaded_docs += 1
                except StopIteration:
                    break
            
            if not batch_raw_texts:
                break

            # 2. تنظيف النصوص قبل الفهرسة والـ Embedding
            try:
                payload = {"texts": batch_raw_texts, **PREPROCESS_FLAGS}
                response = requests.post(preprocess_batch_url(), json=payload)
                response.raise_for_status()
                cleaned_texts = response.json()["results"] 
            except Exception as e:
                print(f"Preprocessing failed, falling back to raw: {e}")
                cleaned_texts = batch_raw_texts

            # 3. استخراج المتجهات من النصوص النظيفة وتمريرها للفهرسة
            # تجميع الكلمات المقطعة (Tokens) لتصبح نصاً كاملاً (String) إذا لزم الأمر
            cleaned_texts_for_embedding = [" ".join(doc) if isinstance(doc, list) else str(doc) for doc in cleaned_texts]
            vectors = embedding_model.encode(cleaned_texts_for_embedding, normalize_embeddings=True).tolist()
            
            # 4. فهرسة النتائج (تمرير النصوص النظيفة والمتجهات الجاهزة)
            self._index_batch(batch_ids, cleaned_texts, vectors)

        # 5. حفظ المانيفست
        manifest = self.builder.save(
            index_dir=index_dir,
            dataset_name=self.dataset_name,
            embedding_model=EMBEDDING_MODEL,
            index_scale_mode=index_scale_mode,
            max_docs_cap=max_docs if max_docs != float("inf") else None,
        )
        print(f"Manifest written: {manifest.get('timestamp')}")

    def _index_batch(self, ids, cleaned_texts, embeddings):
        """إضافة البيانات للفهرس (مباشرة دون طلب Preprocessing إضافي)."""
        try:
            count = self.builder.add_documents(ids, cleaned_texts, embeddings)
            print(f"Indexed batch of {count} documents.")
        except Exception as e:
            print(f"Indexing failed in _index_batch: {e}")

def main():
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