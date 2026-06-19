# Task 03 - Indexing

## Requirement Description
Task 3 يطلب بناء index structures مناسبة (مثل inverted index) لتسريع retrieval وضمان كفاءة الوصول للمصطلحات.

## ما الذي تم تنفيذه
- بناء inverted index من preprocessed tokens.
- حفظ عدة artifacts على القرص:
  - `vsm_index.json`
  - `bm25_index.json`
  - `embeddings_index.json`
  - `metadata.json`
- تخزين BM25 postings بصيغة تدعم runtime dynamic scoring.
- حفظ corpus statistics داخل metadata (`total_docs`, `avg_doc_len`, `doc_lengths`, `idf_weights`).

## الملفات والمكونات المرتبطة
- `indexing_service/app/core/indexer.py`
- `indexing_service/app/main.py`
- مسارات التخزين:
  - `index_data/`
  - `indexing_service/index_data/` (يوجد تكرار في المسار)

## Algorithms and Data Structures
- Inverted index: term -> doc -> term frequency.
- TF-IDF weights لـ VSM.
- BM25-ready postings.
- Dense vector store محفوظ كـ JSON.

## Inputs and Outputs
- Input: docs من `ir_datasets` + preprocessing API output + embedding vectors.
- Output: JSON index files التي يعتمد عليها `retrieval_service`.

## Design Decisions and Data Flow
1. قراءة docs من dataset (مع `max_docs` cap في التنفيذ الحالي).
2. توليد embeddings لكل document text.
3. إرسال النصوص إلى preprocessing service على شكل Batches.
4. بناء postings وdoc stats.
5. حساب الأوزان النهائية وحفظ artifacts.

## تقييم جودة التنفيذ من منظور IR
- **هل dataset size مناسب؟** التصميم النظري جيد، لكن JSON-based full-load indexing غير مناسب عملياً لأحجام كبيرة جداً.
- **هل split/preparation صحيحة؟** لا يوجد دعم واضح لمسار test queries/qrels في indexing workflow.
- **هل algorithms مناسبة؟** نعم كبنية أساسية (inverted + BM25 stats).
- **هل metrics meaningful؟** لا يوجد indexing QA metrics واضحة (coverage/OOV/empty rate).
- **هل يوجد flaws أو leakage أو overfitting risks؟**
  - `max_docs=2000` في script قد يمنع indexing كامل dataset.
  - وجود index directories متعددة قد يسبب mismatch بين ما يُبنى وما يُقرأ.
  - لا يوجد index manifest/versioning شامل.
- **التوافق مع IR best practices؟** مقبول كبداية Prototype، ويحتاج تحسينات scalability/reproducibility.

## Observations and Recommendations
- اعتماد storage layer أنسب للأحجام الكبيرة (بدل JSON الخام).
- توحيد index path في مكان واحد فقط.
- إضافة index manifest يحوي dataset id + preprocessing config + model info + timestamp.
- إضافة sanity checks بعد البناء (doc counts, missing embeddings, empty tokens).
