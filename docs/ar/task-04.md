# Task 04 - Query Processing

## Requirement Description
Task 4 يطلب معالجة user query بنفس preprocessing logic المستخدمة للوثائق، ثم تمثيلها/مطابقتها بأسلوب متوافق مع document representation.

## ما الذي تم تنفيذه
- `retrieval_service` يرسل query إلى `preprocessing_service` عبر `/preprocess` باستخدام نفس الإعدادات العامة.
- بعد preprocessing يتم routing حسب `representation_mode`:
  - `vsm`
  - `bm25`
  - `embedding`
  - `hybrid_parallel`
  - `hybrid_serial`
- معاملات BM25 (`k1`, `b`) تمر ديناميكياً من request وUI.

## الملفات والمكونات المرتبطة
- `retrieval_service/app/main.py`
- `retrieval_service/app/core/search_engine.py`
- `preprocessing_service/app/main.py`
- `preprocessing_service/app/core/cleaner.py`
- `app_ui.py`

## Algorithms and Techniques
- VSM query weighting مع log-TF-IDF.
- BM25 runtime scoring.
- Embedding encoding + cosine similarity.
- Hybrid Parallel (RRF) وHybrid Serial (re-ranking).

## Inputs and Outputs
- Input: raw query + representation mode + optional BM25 params.
- Output: query tokens + ranked result map + total results.

## Design Decisions and Data Flow
1. التحقق من أن query غير فارغة.
2. preprocessing عبر service مستقلة.
3. إعادة تحميل engines من disk لضمان قراءة أحدث index.
4. تنفيذ ranking حسب mode.
5. إعادة النتائج إلى UI.

## تقييم جودة التنفيذ من منظور IR
- **هل dataset size مناسب؟** الخوارزميات مناسبة مبدئياً، لكن full-scan embedding search لن يتوسع جيداً مع dataset كبيرة.
- **هل split/preparation صحيحة؟** preprocessing consistency جيدة نسبياً، لكن لا يوجد benchmark query protocol واضح.
- **هل algorithms مناسبة؟** نعم كbaseline؛ وخاصة مع وجود عدة modes.
- **هل metrics meaningful ومحتسبة صح؟** لا يوجد تكامل فعلي مع MAP/Recall/P@10/nDCG.
- **هل يوجد flaws أو leakage أو overfitting risks؟**
  - VSM cosine غير مكتمل (document norm مفقود).
  - إعادة تهيئة engines كل request تؤثر على latency.
  - serial hybrid candidate pool الافتراضي منخفض جداً.
- **التوافق مع IR best practices؟** متوسط؛ جيد كبداية لكنه يحتاج optimization + تقييم منهجي.

## Observations and Recommendations
- اعتماد cache آمن للفهارس والنماذج بدل reload بكل request.
- استخدام vector index مثل FAISS/HNSW بدل linear scan للـ embeddings.
- إضافة query analytics وedge-case handling.
- ربط retrieval outputs مع evaluation service فور اكتمال metrics pipeline.
