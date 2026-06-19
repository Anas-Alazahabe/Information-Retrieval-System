# شرح التنفيذ — ماذا بُني في المشروع وكيف يعمل الكود

هذا المستند يشرح **ما تم تنفيذه** وفق خطة Tasks 1–4، بطريقة تساعدك كمطور IR لأول مرة على فهم تدفق البيانات والملفات المهمة.

---

## 1) الفكرة العامة

نظام الاسترجاع يمر بمرحلتين منفصلتين:

### أ) وقت الفهرسة (Index Time) — تُنفَّذ نادراً

1. قراءة وثائق من `msmarco-passage` عبر `ir_datasets`
2. توليد **embedding** لكل وثيقة (`sentence-transformers`)
3. إرسال النص الخام لـ **preprocessing** (تنظيف، stopwords، lemmatization)
4. بناء هياكل الفهرس (معكوس، BM25، VSM، metadata)
5. حفظ كل شيء كملفات JSON في `index_data/`

### ب) وقت الاستعلام (Query Time) — تُنفَّذ في كل بحث

1. المستخدم يكتب سؤالاً
2. نفس **preprocessing** يُطبَّق على الاستعلام (مهم للاتساق)
3. **retrieval** يختار محركاً حسب النمط: VSM / BM25 / Embedding / Hybrid
4. إرجاع قائمة `doc_id → score` مرتبة

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  ir_datasets │ ──► │ preprocessing    │ ──► │ IndexBuilder│ ──► index_data/
└─────────────┘     └──────────────────┘     └─────────────┘

┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  User query  │ ──► │ preprocessing    │ ──► │ SearchEngine │ ──► ranked docs
└─────────────┘     └──────────────────┘     └─────────────┘
```

---

## 2) الطبقة المشتركة `shared/`

قبل التنفيذ كانت كل خدمة تستخدم مسارات وإعدادات مختلفة. الآن **`shared/`** هو المصدر الواحد للحقيقة.

### `shared/ir_config.py`

- يحدد `INDEX_DIR` (افتراضياً `index_data/` في جذر المشروع)
- عناوين الخدمات: preprocessing `:8000`، retrieval `:8002`
- إعدادات preprocessing الافتراضية: lemmatization نعم، stemming لا، stopwords نعم
- أوضاع الحجم: `dev` (5K)، `preval` (30K)، `full`
- قائمة أنماط الاسترجاع الصالحة
- `get_git_commit()` لكتابة commit في `index_manifest.json`

### `shared/index_builder.py` — قلب الفهرسة (Task 3)

الكلاس `IndexBuilder`:

| الدالة | الدور |
|--------|-------|
| `add_documents()` | يضيف دفعة وثائق: يبني الفهرس المعكوس الخام، يحفظ embeddings، يتتبع الوثائق الفارغة |
| `_compute_indices()` | يحسب TF-IDF (VSM)، بيانات BM25، و **`doc_norms`** (طول متجه كل وثيقة) |
| `validate()` | فحوصات بعد البناء: عدد الوثائق، embeddings الناقصة، المصطلحات الفريدة |
| `save()` | يكتب 5 ملفات JSON + `index_manifest.json` |

**لماذا `doc_norms` مهم؟**  
في VSM يجب استخدام **Cosine Similarity** = حاصل الضرب / (norm الاستعلام × norm الوثيقة). سابقاً كان الكود يقسم على norm الاستعلام فقط — وهذا خطأ. الآن تُحسب norms وقت الفهرسة وتُخزَّن في `metadata.json`.

### `shared/index_store.py` — قراءة الفهارس (Task 3.4)

- `IndexStore` بروتوكول (واجهة) لقراءة الفهارس
- `JsonIndexStore` التنفيذ الحالي بملفات JSON
- `index_ready()` يتحقق من وجود الملفات
- `get_index_mtime()` لاكتشاف تحديث الفهرس وإعادة التحميل

**الفائدة:** مستقبلاً يمكن استبدال JSON بـ FAISS أو SQLite دون إعادة كتابة محركات البحث.

---

## 3) Task 1 — Preprocessing (`preprocessing_service/`)

### `app/core/cleaner.py` — `TextCleaner`

خط أنابيب المعالجة:

1. إزالة URLs والرموز الزائدة
2. تحويل لحروف صغيرة
3. tokenization
4. إزالة stopwords (NLTK)
5. lemmatization عبر spaCy **أو** fallback تقليدي
6. stemming اختياري (Porter) — **معطّل افتراضياً** في الإعدادات المشتركة

`/health` يُرجع:

- `lemmatization_mode`: `"spacy"` أو `"traditional_fallback"`
- `spacy_available`: true/false

هذا يُسجَّل أيضاً في `index_manifest.json` عند الفهرسة.

### `app/main.py`

- `POST /preprocess` — نص واحد
- `POST /preprocess-batch` — دفعة (يستخدمها الـ indexer)
- القيم الافتراضية تأتي من `PREPROCESS_FLAGS` في `ir_config` لضمان **نفس المعالجة** للوثائق والاستعلامات

---

## 4) Task 3 — Indexing (`indexing_service/`)

### مساران للفهرسة — نفس المنطق الداخلي

| المسار | الملف | الاستخدام |
|--------|-------|-----------|
| **CLI (الموصى به)** | `app/core/indexer.py` | `python -m indexing_service.app.core.indexer --scale dev` |
| **HTTP API** | `app/main.py` | `POST /add-to-index` ثم `POST /save-index` |

`DatasetIndexer` (CLI):

1. يحمّل `ir_datasets.load("msmarco-passage")`
2. لكل دفعة (500 وثيقة): embedding + استدعاء preprocessing batch
3. يمرر النتائج لـ `IndexBuilder.add_documents()`
4. في النهاية `builder.save()` → `index_data/`

---

## 5) Task 2 — Representations (`retrieval_service/app/core/search_engine.py`)

### `BM25SearchEngine`

- يقرأ `bm25_index.json` و `metadata.json`
- لكل مصطلح في الاستعلام: يحسب IDF ودرجة BM25 مع معاملات `k1` و `b` (قابلة للتعديل من الواجهة)

### `EmbeddingSearchEngine`

- يقرأ `embeddings_index.json`
- يحمّل نموذج `all-MiniLM-L6-v2` عند أول استعلام (lazy)
- يحسب cosine similarity بين متجه الاستعلام وكل وثيقة (**full scan** — مناسب للأحجام الصغيرة/المتوسطة)

### `HybridSearchEngine`

**Parallel (`hybrid_parallel`):**

1. يشغّل BM25 و Embedding معاً
2. يدمج القائمتين بـ **RRF** (Reciprocal Rank Fusion)
3. استُبدل `list.index()` O(n²) بقواميس ترتيب O(n)

**Serial (`hybrid_serial`):**

1. BM25 يختار أفضل `top_n_filter` وثيقة (افتراضياً **100** بدلاً من 2 سابقاً)
2. Embedding يعيد ترتيب هذا المجموعة فقط — أسرع وأفضل recall

### VSM — `VSMMatcher` (`retrieval_service/app/core/matching/vsm_matcher.py`)

1. يبني متجه الاستعلام (TF-IDF)
2. يجمع حاصل الضرب مع أوزان الوثائق من `vsm_index.json`
3. يقسم على `query_norm * doc_norm` — **cosine صحيح**

---

## 6) Task 6 — Matching & Ranking (`retrieval_service/app/core/matching/`)

### البنية

```
POST /search → QueryRepresentation → MatcherRegistry → Ranker → results
```

| Matcher | طريقة المطابقة |
|---------|----------------|
| `VSMMatcher` | cosine_similarity |
| `BM25Matcher` | bm25 |
| `EmbeddingMatcher` | cosine_similarity |
| `HybridParallelMatcher` | rrf |
| `HybridSerialMatcher` | bm25_filter_cosine_rerank |

### الأداء

- **NumPy batch** (`IR_EMBEDDING_BACKEND=numpy`) للمجموعات المتوسطة
- **FAISS** (`embeddings.faiss`) عند ≥ `FAISS_THRESHOLD` وثائق
- `GET /matchers` — قائمة الأنماط وطرق المطابقة
- أزمنة منفصلة: `encode_ms`, `match_ms`, `rank_ms`

---

## 7) Task 4 — Query Processing (`retrieval_service/app/main.py`)

### `POST /search` — خط الأنابيب

```
validate mode → preprocess query → match → rank → truncate top_k → JSON response
```

التحسينات المُنفَّذة:

| الميزة | الشرح |
|--------|-------|
| **تحميل الفهرس مرة واحدة** | عند startup + cache لـ VSM |
| **إعادة تحميل ذكية** | إذا تغيّر `mtime` لملفات الفهرس |
| **`POST /reload-index`** | إعادة تحميل يدوية بعد إعادة البناء |
| **التحقق من النمط** | 400 مع قائمة الأنماط الصالحة |
| **503 بدون فهرس** | رسالة واضحة مع مسار `INDEX_DIR` |
| **`timing`** | `preprocess_ms`, `encode_ms`, `match_ms`, `rank_ms`, `total_ms` |

---

## 8) التقييم `evaluation_service/`

### `app/metrics.py`

| المقياس | المعنى |
|---------|--------|
| **P@10** | نسبة الوثائق ذات الصلة في أعلى 10 |
| **Recall** | كم من الوثائق ذات الصلة تم استرجاعها |
| **MAP** | متوسط Average Precision عبر الاستعلامات |
| **nDCG@10** | جودة الترتيب مع خصم المواضع البعيدة |

### `run.py`

- يحمّل استعلامات و qrels من `ir_datasets`
- لكل استعلام: يستدعي `retrieval /search` لكل نمط
- يجمع المقاييس ويحفظ JSON في `evaluation_results/`

---

## 9) الواجهة `app_ui.py`

- Streamlit بالعربية
- اختيار نمط الاسترجاع
- منزلقات `k1` و `b` لـ BM25
- منزلق `top_n_filter` للـ hybrid serial
- يعرض `matching_method` ومعاملات المطابقة
- منزلق `top_k` لعدد النتائج المعروضة

---

## 10) الاختبارات `tests/`

| الملف | ماذا يختبر |
|-------|------------|
| `test_preprocessing.py` | حالات حافة: نص فارغ، URLs، علامات ترقيم |
| `test_matchers.py` | مطابقون مع فهرس ثابت صغير — ترتيب متوقع |
| `test_retrieval_regression.py` | فهرس صغير — تكامل matchers |
| `test_metrics.py` | صحة حساب MAP / nDCG |

---

## 11) ملخص التغييرات الرئيسية (قبل ← بعد)

| المشكلة السابقة | الحل |
|-----------------|------|
| مساران للفهرس (`index_data` vs `indexing_service/index_data`) | مسار واحد عبر `INDEX_DIR` |
| VSM بدون norm الوثيقة | `doc_norms` في metadata + cosine كامل |
| Hybrid serial بـ 2 مرشحين فقط | افتراضي 100 (`IR_SERIAL_TOP_N`) |
| RRF بطيء O(n²) | قواميس ترتيب |
| إعادة تحميل الفهرس كل استعلام | cache + startup load + mtime |
| مساران فهرسة منفصلان | `IndexBuilder` مشترك + CLI + API |
| لا manifest ولا فحوصات | `index_manifest.json` + `sanity_checks` |
| لا تقييم | `evaluation_service` + مقاييس IR + `evaluation_results/` |
| Embedding full scan فقط | NumPy + FAISS عند الحجم الكبير |

---

## 12) ما الذي لم يُحلّ بعد (للعلم)

- **`indexing_service/index_data/`** قديم — يمكن حذفه يدوياً
- **`api_gateway`** ما زال فارغاً
- الفهرسة الكاملة `full` على msmarco تتطلب وقتاً وذاكرة كبيرة
- تقييم HTTP الكامل يتطلب تشغيل preprocessing + retrieval أثناء `evaluation_service.run`

---

## 13) أين تقرأ الكود أولاً؟

ترتيب مقترح للمطور الجديد:

1. `shared/ir_config.py` — الإعدادات
2. `preprocessing_service/app/core/cleaner.py` — كيف يُنظَّف النص
3. `shared/index_builder.py` — كيف تُبنى الفهارس
4. `indexing_service/app/core/indexer.py` — كيف تُقرأ الوثائق
5. `retrieval_service/app/core/matching/` — مطابقة وترتيب (Task 6)
6. `retrieval_service/app/core/search_engine.py` — تحميل الفهارس والتضمين/FAISS
7. `retrieval_service/app/main.py` — ربط كل شيء في API البحث
8. `app_ui.py` — تجربة المستخدم
