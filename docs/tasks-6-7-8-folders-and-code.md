# الطلب السادس والسابع والثامن — الفولدرات والأكواد المرتبطة

هذا الملف يجيب على السؤال: **«هلق الطلب السادس والسابع والتامن شو الفولدرات والأكواد يلي بتخصن؟»**

يعتمد على متطلبات [`IR Project 2026.md`](../IR%20Project%202026.md) وعلى البنية الفعلية في المستودع.

---

## ملخص سريع

| الطلب | الموضوع | أين يتركز التنفيذ في المشروع |
|-------|---------|------------------------------|
| **6** | مطابقة الاستعلام وترتيب النتائج | `retrieval_service/app/core/matching/` + `search_engine.py` |
| **7** | بنية SOA (خدمات مستقلة) | كل مجلد `*_service/` + `shared/` + `app_ui.py` |
| **8** | تقييم النظام (MAP, Recall, P@10, nDCG) | `evaluation_service/` + `scripts/` + `evaluation_results/` |

---

## الطلب السادس — مطابقة الاستعلام وترتيب النتائج (Query Matching & Ranking)

### ماذا يطلب التكليف؟

بناء دالة/نظام يطابق تمثيل الاستعلام مع تمثيل الوثائق ويرتب النتائج حسب أعلى درجات التشابه، مع اعتماد **طريقة مطابقة مناسبة لكل نموذج تمثيل**:

- **VSM / Embedding** → تشابه جيب التمام (Cosine Similarity)
- **BM25** → صيغة BM25
- **Hybrid** → دمج نتائج (مثلاً RRF للمتموازي، أو BM25 ثم إعادة ترتيب دلالي للتسلسلي)

### الفولدرات والأكواد الخاصة بالطلب 6

#### 1) القلب الأساسي — خدمة الاسترجاع

```
retrieval_service/
├── app/
│   ├── main.py                          ← نقطة الدخول: POST /search، GET /matchers
│   └── core/
│       ├── search_engine.py             ← محركات BM25 و Embedding (الحساب الفعلي للدرجات)
│       └── matching/                    ← ★ المجلد الرئيسي للطلب 6
│           ├── __init__.py
│           ├── base.py                  ← QueryRepresentation، MatchParams، Ranker
│           ├── registry.py              ← MatcherRegistry: يوجّه كل mode لـ matcher
│           ├── vsm_matcher.py           ← VSM → cosine similarity
│           ├── bm25_matcher.py          ← BM25 → صيغة BM25
│           ├── embedding_matcher.py     ← Embedding → cosine similarity
│           └── hybrid_matcher.py        ← hybrid_parallel (RRF) + hybrid_serial (BM25 ثم rerank)
```

**دور كل ملف مهم:**

| الملف | الوظيفة |
|-------|---------|
| `main.py` | المسار: `preprocess → match → rank → JSON response` مع توقيت `preprocess_ms`, `match_ms`, `rank_ms` |
| `base.py` | عقود المطابقة والترتيب؛ `Ranker.rank()` يفرز تنازلياً مع كسر التعادل بـ `doc_id` |
| `registry.py` | يربط `representation_mode` بالـ matcher المناسب |
| `search_engine.py` | `BM25SearchEngine.search()` و `EmbeddingSearchEngine` (loop / numpy / FAISS) |
| `vsm_matcher.py` | يبني متجه الاستعلام log-TF-IDF ويحسب cosine كامل (query norm × doc norm) |
| `hybrid_matcher.py` | متوازي: RRF؛ تسلسلي: أفضل N من BM25 ثم cosine rerank |

#### 2) إعدادات ووصف طرق المطابقة

```
shared/
├── ir_config.py          ← MATCHER_METADATA، RRF_K، SERIAL_HYBRID_TOP_N، FAISS_THRESHOLD
├── index_store.py        ← قراءة vsm_index، bm25_index، embeddings من index_data/
└── index_builder.py      ← بناء FAISS عند الفهرسة (يدعم مطابقة embedding على مقياس كبير)
```

#### 3) بيانات الفهرسة (مدخلات المطابقة، تُبنى في الطلب 3 لكن تُستهلك هنا)

```
index_data/
├── vsm_index.json
├── bm25_index.json
├── embeddings_index.json
├── embeddings.faiss              ← عند ≥ 10K وثيقة
├── embeddings_id_map.json
├── metadata.json
└── index_manifest.json
```

#### 4) اختبارات الطلب 6

```
tests/
├── test_matchers.py              ← اختبارات وحدة لكل matcher
├── test_retrieval_regression.py  ← اختبارات انحدار على API الاسترجاع
└── fixtures/matcher_index/       ← فهرس صغير ثابت للاختبار
```

#### 5) واجهة المستخدم (عرض طريقة المطابقة)

```
app_ui.py                         ← يعرض matching_method، params، top_k من استجابة /search
```

#### 6) سكربتات مساعدة للتحقق من المطابقة

```
scripts/run_matcher_eval_smoke.py ← تقييم سريع يمر عبر matchers
```

### تدفق التنفيذ (الطلب 6)

```
استعلام المستخدم
    → retrieval_service: POST /search
        → preprocessing_service: POST /preprocess  (معالجة الاستعلام — طلب 4)
        → MatcherRegistry.get(mode).match(query_repr, params)
        → Ranker.rank(scores, top_k)
    → نتائج مرتبة { doc_id: score }
```

### Endpoints مرتبطة بالطلب 6

| Endpoint | الخدمة | الغرض |
|----------|--------|-------|
| `POST /search` | `retrieval_service` (:8002) | البحث والمطابقة والترتيب |
| `GET /matchers` | `retrieval_service` | قائمة الأنماط وطرق المطابقة |
| `POST /reload-index` | `retrieval_service` | إعادة تحميل الفهارس بعد إعادة البناء |

### أوامر تشغيل مفيدة

```powershell
# اختبارات المطابقة
python -m pytest tests/test_matchers.py -q

# قائمة طرق المطابقة
# GET http://127.0.0.1:8002/matchers
```

### توثيق إضافي في المشروع

- [`docs/task-06.md`](task-06.md)
- [`docs/task-06-implementation-plan.md`](task-06-implementation-plan.md)

---

## الطلب السابع — بنية SOA (Service Oriented Architecture)

### ماذا يطلب التكليف؟

تصميم النظام كمجموعة **خدمات مستقلة**، كل خدمة مسؤولة عن مهمة محددة، مع:

- فصل واضح للمسؤوليات
- تواصل بين الخدمات (REST API)
- إمكانية تشغيل/اختبار كل خدمة لوحدها
- كود منظم قابل للصيانة والتوسع
- مخطط معماري في التقرير يوضح البنية والتواصل

### الفولدرات والأكواد الخاصة بالطلب 7

#### 1) الخدمات المستقلة (Microservices)

```
preprocessing_service/          ← خدمة معالجة البيانات (الطلب 1)
├── app/
│   ├── main.py                 ← FastAPI: /preprocess، /preprocess-batch، /health
│   └── core/
│       └── cleaner.py          ← tokenization، stopwords، stemming، lemmatization

indexing_service/               ← خدمة الفهرسة (الطلب 3)
├── app/
│   ├── main.py                 ← FastAPI: /add-to-index، /save-index، /health
│   └── core/
│       └── indexer.py          ← CLI: python -m indexing_service.app.core.indexer

retrieval_service/              ← خدمة البحث والاسترجاع + المطابقة (الطلب 4 + 6)
├── app/
│   ├── main.py                 ← FastAPI: /search، /matchers، /health
│   └── core/
│       ├── search_engine.py
│       └── matching/

query_refinement_service/       ← خدمة تحسين الاستعلام (الطلب 5)
├── app/
│   ├── main.py                 ← FastAPI: /refine، /suggest، /health
│   ├── models.py
│   └── core/
│       ├── refiner.py          ← تنسيق تقنيات التحسين
│       ├── prf.py              ← Pseudo-Relevance Feedback
│       ├── synonym_expander.py
│       ├── history.py          ← سياق البحث السابق
│       └── suggestions.py      ← اقتراحات استعلام (UX)

evaluation_service/             ← خدمة التقييم (الطلب 8 — لكنها جزء من SOA)
├── app/
│   ├── main.py                 ← FastAPI: POST /evaluate، GET /health
│   └── metrics.py
├── run.py                      ← CLI للتقييم
└── run_fixture_eval.py
```

#### 2) الطبقة المشتركة (عقود بين الخدمات)

```
shared/
├── ir_config.py                ← منافذ، URLs، ثوابت، MATCHER_METADATA
├── index_store.py              ← قراءة موحدة لملفات الفهرس
├── index_builder.py            ← بناء الفهرس (يُستدعى من indexing_service)
├── search_pipeline.py          ← تنسيق: refine (اختياري) → search
└── query_suggestions.py        ← مساعد لاقتراحات الاستعلام
```

#### 3) الواجهة الأمامية / بوابة الاستخدام (UI Gateway)

```
app_ui.py                       ← Streamlit: يتصل بـ retrieval + refinement عبر HTTP
.streamlit/config.toml
```

> **ملاحظة:** مجلد `api_gateway/` مذكور في بعض الوثائق كجزء مخطط له، لكنه **غير منفّذ فعلياً** في الكود الحالي. دور الـ API Gateway يقوم به عملياً:
> - `app_ui.py` للمستخدم
> - `shared/search_pipeline.py` للتنسيق البرمجي بين refinement و retrieval
> - `evaluation_service` عند التقييم الجماعي

#### 4) سكربتات التشغيل والبناء

```
scripts/
├── build_query_suggestion_index.py
├── bootstrap_index_manifest.py
├── run_refinement_ablation.py
└── run_matcher_eval_smoke.py

requirements.txt                ← تبعيات المشروع
```

#### 5) مخزن البيانات المشترك بين الخدمات

```
index_data/                     ← مخرجات indexing_service، مدخلات retrieval_service
```

### خريطة SOA — المنافذ والتواصل

| الخدمة | المنفذ الافتراضي | بروتوكول | مسؤولية |
|--------|------------------|----------|---------|
| `preprocessing_service` | 8000 | REST | معالجة النصوص والاستعلامات |
| `retrieval_service` | 8002 | REST | بحث، مطابقة، ترتيب |
| `query_refinement_service` | 8003 | REST | تحسين الاستعلام واقتراحات |
| `indexing_service` | (CLI/API) | REST + CLI | بناء وحفظ الفهارس |
| `evaluation_service` | (CLI/API) | REST + CLI | حساب مقاييس IR |
| `app_ui.py` | Streamlit | HTTP client | واجهة المستخدم |

**متغيرات البيئة** (في `shared/ir_config.py`):

- `IR_PREPROCESS_URL` → `http://127.0.0.1:8000`
- `IR_RETRIEVAL_URL` → `http://127.0.0.1:8002`
- `IR_REFINEMENT_URL` → `http://127.0.0.1:8003`
- `IR_INDEX_DIR` → مسار `index_data/`

### مخطط التواصل (مبسّط)

```
                    ┌─────────────────┐
                    │    app_ui.py    │
                    │   (Streamlit)   │
                    └────────┬────────┘
                             │ HTTP
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
   ┌──────────────────┐ ┌─────────────┐ ┌─────────────────────┐
   │ query_refinement   │ │ retrieval   │ │ evaluation_service  │
   │    _service        │ │  _service   │ │  (batch eval)       │
   │     :8003          │ │   :8002     │ │                     │
   └─────────┬──────────┘ └──────┬──────┘ └──────────┬──────────┘
             │                   │                    │
             │                   │ reads              │ calls /search
             │                   ▼                    │
             │            ┌─────────────┐             │
             │            │ index_data/ │◄────────────┘
             │            └──────▲──────┘
             │                   │ writes
             │            ┌──────┴──────┐
             │            │  indexing   │
             │            │  _service   │
             │            └──────┬──────┘
             │                   │ batch preprocess
             │                   ▼
             │            ┌──────────────────┐
             └───────────►│ preprocessing    │
                          │    _service      │
                          │      :8000       │
                          └──────────────────┘
```

### Design Patterns مستخدمة (للتقرير)

| النمط | أين يظهر |
|-------|----------|
| **Microservices / SOA** | كل `*_service/` خدمة FastAPI مستقلة |
| **Registry** | `MatcherRegistry` في الطلب 6 |
| **Protocol / Interface** | `IndexStore`, `BaseMatcher` |
| **Pipeline** | `search_pipeline.search_with_optional_refinement` |
| **Shared Configuration** | `shared/ir_config.py` كمصدر واحد للحقيقة |

### أوامر تشغيل الخدمات (SOA)

```powershell
# 1) معالجة
cd preprocessing_service
uvicorn app.main:app --host 127.0.0.1 --port 8000

# 2) استرجاع
cd retrieval_service
uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload

# 3) تحسين استعلام (اختياري)
cd query_refinement_service
uvicorn app.main:app --host 127.0.0.1 --port 8003 --reload

# 4) واجهة
streamlit run app_ui.py
```

### توثيق إضافي

- [`docs/README.md`](README.md) — نظرة معمارية
- [`docs/developer-guide.md`](developer-guide.md) — تشغيل الخدمات
- [`docs/project-notes.md`](project-notes.md) — فجوات SOA (مثل `api_gateway` غير مكتمل)

---

## الطلب الثامن — تقييم النظام (Evaluation)

### ماذا يطلب التكليف؟

تقييم أداء نظام IR بمقاييس قياسية **لكل نموذج تمثيل ولكل dataset**:

- **MAP** (Mean Average Precision)
- **Recall**
- **Precision@10**
- **nDCG**

ويجب التقييم في حالتين:

1. **قبل** تطبيق الميزات الإضافية (baseline)
2. **بعد** تطبيق الميزات الإضافية (مثل query refinement)

مع تحليل في التقرير: تأثير كل تمثيل، مقارنة النماذج، مساهمة الميزات الإضافية.

### الفولدرات والأكواد الخاصة بالطلب 8

#### 1) خدمة التقييم

```
evaluation_service/
├── app/
│   ├── main.py                 ← ★ run_evaluation() + POST /evaluate
│   └── metrics.py              ← ★ MAP, Recall, P@k, nDCG
├── run.py                      ← CLI: python -m evaluation_service.run
└── run_fixture_eval.py         ← تقييم على fixture صغير
```

**دور الملفات:**

| الملف | الوظيفة |
|-------|---------|
| `metrics.py` | `precision_at_k`, `recall_at_k`, `average_precision`, `ndcg_at_k`, `aggregate_metrics` |
| `main.py` | يحمّل queries/qrels من `ir_datasets`، يستدعي `retrieval_service` لكل استعلام وmode، يجمع المقاييس |
| `run.py` | واجهة سطر أوامر؛ يحفظ JSON في `evaluation_results/` |

#### 2) تنسيق التقييم مع الميزات الإضافية

```
shared/
└── search_pipeline.py          ← search_with_optional_refinement (baseline vs refined)

scripts/
├── run_refinement_ablation.py  ← ★ تقييم before/after: baseline + كل تقنية + combined
└── run_matcher_eval_smoke.py   ← smoke test سريع
```

#### 3) اختبارات المقاييس

```
tests/
└── test_metrics.py             ← unit tests لصحة حساب MAP, P@10, nDCG, ...
```

#### 4) مخرجات التقييم (تقارير جاهزة)

```
evaluation_results/             ← مخرجات CLI الافتراضية
reports/                        ← تقارير سابقة محفوظة
evaluation_report.ipynb         ← notebook لتحليل النتائج
```

أمثلة على أسماء ملفات التقرير:

- `eval_baseline_dev_*.json` — قبل الميزات الإضافية
- `eval_refined_query_preprocess-prf-synonyms_*.json` — بعد التحسين
- `refinement_ablation_summary_*.json` — مقارنة deltas مقابل baseline

#### 5) خدمات يعتمد عليها التقييم (ليست «تقييماً» لكنها ضرورية للتشغيل)

```
retrieval_service/              ← مصدر النتائج المرتبة عبر POST /search
query_refinement_service/       ← عند --use-refinement
preprocessing_service/          ← مطلوبة لأن retrieval يعالج الاستعلام عبرها
index_data/                     ← فهرس يجب أن يكون جاهزاً
```

**Dataset للتقييم:** `msmarco-passage/dev` (من `ir_datasets`، مُعرّف في `shared/ir_config.py` كـ `EVAL_DATASET_NAME`).

### تدفق التقييم

```
evaluation_service.run_evaluation()
    │
    ├─ ir_datasets: تحميل queries + qrels
    │
    └─ لكل representation_mode (vsm, bm25, embedding, hybrid_*):
           لكل query:
               ├─ [baseline]  POST retrieval /search
               └─ [refined]   search_pipeline → refine ثم search
               evaluate_ranked_list(ranked_docs, qrels, k=10)
           aggregate_metrics() → MAP, Recall, P@10, nDCG
    │
    └─ حفظ JSON في evaluation_results/
```

### أوامر التقييم

```powershell
# baseline (قبل الميزات الإضافية)
python -m evaluation_service.run --scale dev --max-queries 20

# بعد query refinement
python -m evaluation_service.run --scale preval --max-queries 50 --use-refinement --refinement-techniques query_preprocess,prf,synonyms

# ablation كامل (before + after لكل تقنية)
python scripts/run_refinement_ablation.py --scale preval --max-queries 50
```

**شرط التشغيل:** الخدمات `preprocessing` (8000)، `retrieval` (8002)، وعند التحسين `refinement` (8003) يجب أن تكون شغّالة.

### API التقييم

| Endpoint | الغرض |
|----------|-------|
| `POST /evaluate` | تشغيل تقييم كامل عبر HTTP |
| `GET /health` | فحص جاهزية خدمة التقييم |

---

## جدول مرجعي: أي طلب يمس أي مجلد؟

| المجلد / الملف | طلب 6 | طلب 7 | طلب 8 |
|----------------|:-----:|:-----:|:-----:|
| `retrieval_service/app/core/matching/` | ●●● | ● | ● (يُستدعى) |
| `retrieval_service/app/core/search_engine.py` | ●●● | ● | ● |
| `retrieval_service/app/main.py` | ●● | ●● | ● |
| `preprocessing_service/` | ○ (مدخل) | ●●● | ● (شرط تشغيل) |
| `indexing_service/` | ○ (فهرس) | ●●● | ○ |
| `query_refinement_service/` | ○ | ●●● | ● (تقييم after) |
| `evaluation_service/` | ○ | ●● | ●●● |
| `shared/` | ●● | ●●● | ●● |
| `app_ui.py` | ● | ●● | ○ |
| `index_data/` | ●● | ●● | ● |
| `tests/test_matchers.py` | ●●● | ○ | ○ |
| `tests/test_metrics.py` | ○ | ○ | ●●● |
| `scripts/run_refinement_ablation.py` | ○ | ● | ●●● |
| `evaluation_results/` / `reports/` | ○ | ○ | ●●● |
| `api_gateway/` | — | ○ (غير منفّذ) | — |

**الرموز:** ●●● أساسي | ●● مهم | ● مرتبط | ○ غير مباشر

---

## إجابة مباشرة على السؤال

### الطلب 6 — شو الفولدرات والأكواد؟

**الفولدر الأساسي:** `retrieval_service/app/core/matching/`

**أهم الملفات:**
- `retrieval_service/app/main.py` — `/search` و `/matchers`
- `retrieval_service/app/core/search_engine.py` — BM25 و Embedding
- `shared/ir_config.py` — وصف كل matcher
- `tests/test_matchers.py` — اختبارات

### الطلب 7 — شو الفولدرات والأكواد؟

**الفولدرات = الخدمات:**
- `preprocessing_service/`
- `indexing_service/`
- `retrieval_service/`
- `query_refinement_service/`
- `evaluation_service/`
- `shared/` — العقود المشتركة
- `app_ui.py` — الواجهة

**التواصل:** REST بين الخدمات؛ المنافذ في `shared/ir_config.py`.

### الطلب 8 — شو الفولدرات والأكواد؟

**الفولدر الأساسي:** `evaluation_service/`

**أهم الملفات:**
- `evaluation_service/app/metrics.py` — حساب MAP, Recall, P@10, nDCG
- `evaluation_service/app/main.py` — `run_evaluation()`
- `evaluation_service/run.py` — CLI
- `scripts/run_refinement_ablation.py` — before/after
- `tests/test_metrics.py`
- `evaluation_results/` و `reports/` — مخرجات التقارير

---

## ملاحظات للتقرير النهائي

1. **الطلب 6 و 7 متداخلان:** المطابقة والترتيب منفّذة داخل `retrieval_service` وهي خدمة SOA مستقلة.
2. **الطلب 8 يعتمد على 6:** التقييم يقيس جودة مخرجات `POST /search` لكل `representation_mode`.
3. **`api_gateway`:** مذكور في التكليف لكن غير مبني؛ اذكروا في التقرير أن `app_ui.py` + `search_pipeline.py` يقومان بالتنسيق.
4. **Dataset:** التقييم يستخدم `ir_datasets` و qrels؛ تأكدوا من تشغيل الفهرسة على نفس المجموعة قبل التقييم.

---

*آخر تحديث بناءً على حالة المستودع ووثائق `docs/` (باستثناء `docs/ar/`).*
