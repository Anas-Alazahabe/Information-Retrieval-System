# Task 02 - Document Representation

## Requirement Description
Task 2 يطلب تمثيل الوثائق باستخدام:
- VSM TF-IDF
- Embeddings
- BM25
- Hybrid representation بنمطين:
  - Parallel (مع Fusion/Scoring method)
  - Serial
مع إتاحة معاملات BM25 وخيار hybrid في UI أو التقرير.

## ما الذي تم تنفيذه
- توليد أوزان VSM TF-IDF وحفظها ضمن index artifacts.
- حفظ BM25 postings بصيغة `tf` و`doc_len` لحساب dynamic scoring وقت query.
- توليد embeddings عبر `SentenceTransformer("all-MiniLM-L6-v2")`.
- تنفيذ Hybrid retrieval:
  - Parallel: دمج BM25 + Embedding عبر Reciprocal Rank Fusion (RRF).
  - Serial: BM25 candidate filtering ثم Embedding re-ranking.
- UI تدعم التبديل بين modes وتغيير `k1`, `b`.

## الملفات والمكونات المرتبطة
- `indexing_service/app/core/indexer.py`
- `indexing_service/app/main.py`
- `retrieval_service/app/core/search_engine.py`
- `retrieval_service/app/main.py`
- `app_ui.py`
- `test_vsm.py`

## Algorithms, Models, and Techniques
- VSM:
  - `idf = log10(N/df)`
  - `doc_weight = (1 + log10(tf)) * idf`
  - query weighting بنفس الفكرة.
- BM25:
  - dynamic scoring مع معاملات `k1`, `b`.
- Embedding:
  - dense vector encoding عبر MiniLM.
  - cosine similarity للمطابقة.
- Hybrid:
  - Parallel fusion via RRF.
  - Serial two-stage retrieval/re-ranking.

## Inputs and Outputs
- Inputs: processed tokens (lexical) + raw text (embedding) + retrieval mode + BM25 params.
- Outputs: ranked `{doc_id: score}`.

## تقييم جودة التنفيذ من منظور IR
- **هل dataset size مناسب؟** الخوارزميات نفسها مناسبة، لكن البيانات المفهرسة الحالية demo-scale.
- **هل split/preparation صحيحة؟** اعتماد one dataset مقبول بسبب supervisor approval، لكن protocol واضح للتقسيم/التقييم غير ظاهر.
- **هل algorithms مناسبة؟** نعم، مزيج lexical + dense + hybrid مناسب جداً كبداية.
- **هل metrics meaningful ومُحتسبة بشكل صحيح؟** غير موصولة بعد مع evaluation pipeline.
- **هل يوجد flaws أو leakage أو overfitting risks؟**
  - لا يوجد leakage واضح في formulas.
  - VSM cosine غير مكتمل (document norm غير محسوب).
  - `top_n_filter=2` في serial منخفض جداً وقد يضر recall.
- **التوافق مع IR principles؟** جيد من ناحية الفكرة، ويحتاج calibration + benchmarking.

## Observations and Recommendations
- تنفيذ full cosine normalization في VSM.
- رفع/تجريب `top_n_filter` بقيم أكبر حسب حجم dataset.
- إضافة Fusion methods إضافية للمقارنة (مثلاً weighted score fusion).
- حفظ model/config metadata لضمان reproducibility.
