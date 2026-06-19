# نظرة عامة تقنية - IR Project 2026

## أهداف المشروع
- بناء Information Retrieval system مبني بأسلوب Service-Oriented Architecture باستخدام Python.
- دعم أكثر من أسلوب Document/Query Representation (VSM TF-IDF, BM25, Embeddings, Hybrid).
- معالجة User Query بلغة طبيعية وإرجاع ranked document IDs.
- توفير UI بسيط يسمح بتغيير retrieval mode والتحكم بمعاملات BM25.

## المعمارية الحالية
- `preprocessing_service`: خدمة FastAPI لتنظيف النصوص، tokenization، stopword removal، مع خيار stemming/lemmatization.
- `indexing_service`: بناء وحفظ index artifacts (`vsm_index`, `bm25_index`, `embeddings_index`, `metadata`).
- `retrieval_service`: تحميل index artifacts وتنفيذ query-time scoring/ranking حسب mode المختار.
- `app_ui.py`: واجهة Streamlit للتعامل مع retrieval API.
- أجزاء موجودة لكن غير مكتملة عملياً حتى الآن: `api_gateway`, `evaluation_service`.

## Technologies and Libraries المستخدمة
- Core: Python, FastAPI, Pydantic, Requests, JSON.
- NLP: NLTK (stopwords, Porter stemmer), spaCy (lemmatization إذا كان model متوفراً).
- Semantic retrieval: `sentence-transformers` مع model `all-MiniLM-L6-v2`.
- UI: Streamlit.
- Dataset access: `ir_datasets` (مُستخدم في indexing path).

## Dataset Overview (حسب التنفيذ الفعلي)
- نص assignment الأصلي يطلب **two datasets** من `ir-datasets.com` بحجم كبير (مذكور >200K docs) مع test queries/qrels.
- ضمن المشروع الحالي، يوجد supervisor-approved exception يسمح باستخدام **one dataset**.
- الكود يحتوي مسار `ir_datasets.load(dataset_name)` مع ذكر `msmarco-passage`.
- artifacts الموجودة حالياً في المستودع صغيرة جداً (3-6 docs) وتبدو demo/toy وليست benchmark-scale.
- لا يوجد حتى الآن qrels-based evaluation pipeline مكتمل في الملفات الظاهرة.

## Execution Workflow
1. تشغيل `preprocessing_service` (`/preprocess`, `/preprocess-batch`).
2. تشغيل indexing logic (`DatasetIndexer`) لتنفيذ:
   - تحميل docs من dataset.
   - توليد embeddings.
   - استدعاء preprocessing batch API.
   - بناء inverted structures و metadata.
   - حفظ `index_data/*.json`.
3. تشغيل `retrieval_service`:
   - استقبال query و mode.
   - preprocessing للـ query عبر preprocessing API.
   - dispatch إلى VSM/BM25/Embedding/Hybrid engines.
4. تشغيل Streamlit UI وإرسال طلبات البحث إلى retrieval service.

## Data and Control Flow
- **Index-time flow**: raw document text -> preprocessing service -> cleaned tokens -> inverted index/BM25 stats + embeddings -> JSON artifacts.
- **Query-time flow**: raw query -> preprocessing service -> query tokens -> ranking engine حسب mode -> sorted scores by document ID.
- **Hybrid parallel**: دمج ranked lists من BM25 و embedding باستخدام RRF.
- **Hybrid serial**: استرجاع أولي عبر BM25 ثم re-ranking لجزء من النتائج عبر embedding similarity.

## حالة التوافق السريعة (Tasks 1-4)
- Task 1 (preprocessing): منفّذ كخدمة مستقلة وبشكل مقبول كبداية.
- Task 2 (representations): VSM, BM25, Embedding, Hybrid كلها موجودة.
- Task 3 (indexing): الفهارس تُبنى وتُحفظ، لكن ما زالت بحاجة تحسينات للاستقرار/scalability.
- Task 4 (query processing): preprocessing + ranking حسب representation mode موجودة.

## دليل التشغيل للمطورين
- **`docs/ar/developer-guide.md`** — خطوات التشغيل، الفهرسة، الخدمات، استكشاف الأخطاء
- **`docs/ar/implementation-overview.md`** — شرح معماري للكود وما تم تنفيذه

## مراجع داخلية
- `docs/ar/task-01.md`
- `docs/ar/task-02.md`
- `docs/ar/task-03.md`
- `docs/ar/task-04.md`
- `docs/ar/project-notes.md`
