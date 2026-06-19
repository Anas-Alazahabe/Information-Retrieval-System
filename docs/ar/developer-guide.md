# دليل المطور — تشغيل مشروع IR 2026

هذا الملف يشرح **كيف تشغّل المشروع خطوة بخطوة**، ومتى تحتاج لإعادة الفهرسة، وكيف تتحقق أن كل شيء يعمل.

---

## 1) ماذا يفعل المشروع؟ (نظرة سريعة)

المشروع نظام **استرجاع معلومات (Information Retrieval)** مبني كخدمات صغيرة:

| المرحلة | ماذا يحدث |
|---------|-----------|
| **الفهرسة (مرة أو عند تغيير البيانات)** | قراءة وثائق من `msmarco-passage` → تنظيف النص → بناء فهارس VSM و BM25 و Embeddings → حفظها في `index_data/` |
| **البحث (كل مرة تستخدم النظام)** | استعلام المستخدم → تنظيف الاستعلام → ترتيب الوثائق حسب النمط المختار (VSM / BM25 / Embedding / Hybrid) |

```
الوثائق الخام  →  preprocessing  →  indexing  →  index_data/*.json
الاستعلام      →  preprocessing  →  retrieval   →  قائمة وثائق مرتبة
```

---

## 2) المتطلبات

- **Python 3.10+** (يفضّل 3.11)
- اتصال إنترنت لأول تشغيل (تحميل NLTK، `ir_datasets`، نموذج `sentence-transformers`)
- **4 نوافذ طرفية** للتشغيل اليومي (أو 3 إذا تخطيت واجهة Streamlit)
- مساحة قرص كافية: وضع `dev` (~5K وثيقة) خفيف؛ وضع `full` ثقيل جداً

### اختياري (لتحسين lemmatization)

```powershell
python -m spacy download en_core_web_sm
```

بدون spaCy يعمل النظام بـ **fallback تقليدي** — لن يتوقف.

---

## 3) الإعداد لأول مرة (مرة واحدة لكل جهاز)

افتح PowerShell من **جذر المشروع**:

```powershell
cd C:\Users\Golden\Documents\ir_core_project

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

تحقق من التثبيت:

```powershell
python -c "import fastapi, ir_datasets, sentence_transformers; print('OK')"
pytest tests/ -q
```

---

## 4) هل التشغيل نفسه كل مرة؟

| الحالة | ماذا تفعل |
|--------|-----------|
| **أول مرة على الجهاز** | إعداد §3 → بناء الفهرس §5 → تشغيل الخدمات §6 |
| **كل يوم للبحث فقط** | شغّل الخدمات §6 + الواجهة — **لا تعيد الفهرسة** |
| **بعد إعادة بناء الفهرس** | أعد تشغيل `retrieval_service` أو استدعِ `POST /reload-index` |
| **بعد تغيير الكود في preprocessing/retrieval** | أعد تشغيل الخدمة المتأثرة فقط |

**القاعدة:** الفهرسة عملية **بطيئة** (دقائق إلى ساعات حسب `--scale`). البحث عملية **سريعة** (ثوانٍ).

---

## 5) بناء الفهرس (Index Build)

### الخطوة أ — شغّل خدمة المعالجة أولاً

الفهرسة تستدعي preprocessing عبر HTTP. **يجب** أن تكون الخدمة شغالة قبل الفهرسة.

```powershell
cd C:\Users\Golden\Documents\ir_core_project\preprocessing_service
..\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

تحقق: افتح `http://127.0.0.1:8000/health`

### الخطوة ب — نفّذ الفهرسة من جذر المشروع

في **نافذة طرفية جديدة**:

```powershell
cd C:\Users\Golden\Documents\ir_core_project
.\.venv\Scripts\Activate.ps1

# وضع التطوير: 5,000 وثيقة (موصى به للبداية)
python -m indexing_service.app.core.indexer --scale dev

# أوضاع أخرى:
# python -m indexing_service.app.core.indexer --scale preval   # 30,000 وثيقة
# python -m indexing_service.app.core.indexer --scale full     # بدون حد (بطيء جداً)
```

### مخرجات الفهرسة

تُحفظ في **`index_data/`** بجذر المشروع (المسار الرسمي الوحيد):

| الملف | المحتوى |
|-------|---------|
| `vsm_index.json` | أوزان TF-IDF لكل مصطلح/وثيقة |
| `bm25_index.json` | إحصاءات BM25 (tf، طول الوثيقة) |
| `embeddings_index.json` | متجهات دلالية لكل وثيقة |
| `metadata.json` | إجمالي الوثائق، IDF، `doc_norms` لـ VSM |
| `index_manifest.json` | سجل البناء: الداتا سِت، الوقت، فحوصات السلامة |

> **تنبيه:** المجلد القديم `indexing_service/index_data/` لم يعد المسار الرسمي — تجاهله.

---

## 6) تشغيل الخدمات للبحث اليومي

تحتاج **3 خدمات** + واجهة اختيارية. كل خدمة في نافذة منفصلة.

### نافذة 1 — Preprocessing (منفذ 8000)

```powershell
cd C:\Users\Golden\Documents\ir_core_project\preprocessing_service
..\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### نافذة 2 — Retrieval (منفذ 8002)

```powershell
cd C:\Users\Golden\Documents\ir_core_project\retrieval_service
..\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload
```

تحقق: `http://127.0.0.1:8002/health` — يجب أن يظهر `"index_files_detected": true`

### نافذة 3 — واجهة Streamlit

```powershell
cd C:\Users\Golden\Documents\ir_core_project
.\.venv\Scripts\Activate.ps1
streamlit run app_ui.py
```

يفتح المتصفح تلقائياً (عادة `http://localhost:8501`).

### خدمة اختيارية — Indexing API (منفذ 8001)

للفهرسة اليدوية عبر HTTP (نادراً ما تحتاجها — الـ CLI أسهل):

```powershell
cd C:\Users\Golden\Documents\ir_core_project\indexing_service
..\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

---

## 7) اختبار سريع بدون الواجهة

```powershell
curl -X POST http://127.0.0.1:8002/search `
  -H "Content-Type: application/json" `
  -d '{"query": "hospital system", "representation_mode": "bm25"}'
```

أنماط الاسترجاع المدعومة:

- `vsm`
- `bm25`
- `embedding`
- `hybrid_parallel`
- `hybrid_serial`

---

## 8) التقييم (Evaluation)

بعد تشغيل preprocessing + retrieval:

```powershell
cd C:\Users\Golden\Documents\ir_core_project
.\.venv\Scripts\Activate.ps1

python -m evaluation_service.run --scale dev --max-queries 20
```

التقرير يُحفظ في `reports/eval_dev_<timestamp>.json` ويتضمن MAP، Recall، P@10، nDCG@10 لكل نمط.

تقييم على فهرس الاختبار الصغير (بدون msmarco):

```powershell
python -m evaluation_service.run_fixture_eval
```

---

## 9) متغيرات البيئة (اختياري)

| المتغير | الافتراضي | الوصف |
|---------|-----------|-------|
| `IR_INDEX_DIR` | `{project}/index_data` | مسار الفهارس |
| `IR_PREPROCESS_URL` | `http://127.0.0.1:8000` | رابط preprocessing |
| `IR_RETRIEVAL_URL` | `http://127.0.0.1:8002` | رابط retrieval |
| `IR_DATASET` | `msmarco-passage` | داتا سِت الفهرسة |
| `IR_INDEX_SCALE` | `dev` | `dev` / `preval` / `full` |
| `IR_MAX_DOCS` | حسب `--scale` | تجاوز يدوي لعدد الوثائق |
| `IR_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | نموذج التضمين |
| `IR_SERIAL_TOP_N` | `100` | مرشحو BM25 في Hybrid Serial |

مثال:

```powershell
$env:IR_INDEX_SCALE = "preval"
python -m indexing_service.app.core.indexer
```

---

## 10) استكشاف الأخطاء

| المشكلة | الحل المحتمل |
|---------|--------------|
| `Index artifacts not found` | نفّذ الفهرسة (§5) أو تحقق من `IR_INDEX_DIR` |
| `Failed to reach preprocessing service` | شغّل preprocessing على المنفذ 8000 |
| `index_files_detected: false` | تأكد أن `index_data/` يحتوي الملفات الخمسة |
| نتائج فارغة بعد الفهرسة | جرّب استعلاماً أبسط؛ تحقق من التوكنز في الاستجابة |
| الفهرسة بطيئة جداً | استخدم `--scale dev` أولاً |
| `spaCy unavailable` | طبيعي بدون النموذج — أو ثبّت `en_core_web_sm` |
| بعد إعادة الفهرسة نتائج قديمة | `POST http://127.0.0.1:8002/reload-index` أو أعد تشغيل retrieval |

---

## 11) هيكل المشروع للمطور الجديد

```
ir_core_project/
├── shared/                    # إعدادات ومكتبة فهرسة مشتركة
│   ├── ir_config.py           # مسارات، منافذ، ثوابت
│   ├── index_builder.py       # بناء VSM/BM25/embeddings
│   └── index_store.py         # قراءة ملفات JSON
├── preprocessing_service/     # تنظيف النص (Task 1)
├── indexing_service/          # CLI + API للفهرسة (Task 3)
├── retrieval_service/         # بحث وترتيب (Task 2 + 4)
├── evaluation_service/        # مقاييس IR (MAP, nDCG, ...)
├── index_data/                # مخرجات الفهرسة (لا تُعدّل يدوياً)
├── tests/                     # اختبارات pytest
├── app_ui.py                  # واجهة Streamlit
└── requirements.txt
```

لشرح معماري أعمق لما تم تنفيذه، راجع:

- `docs/ar/implementation-overview.md` — شرح الكود والتغييرات
- `docs/ar/task-01.md` … `task-04.md` — تفاصيل كل مهمة

---

## 12) قائمة تحقق سريعة

- [ ] `pip install -r requirements.txt`
- [ ] preprocessing يعمل على `:8000`
- [ ] `index_data/` يحتوي الفهارس
- [ ] retrieval يعمل على `:8002` و `/health` يُظهر فهارس جاهزة
- [ ] `streamlit run app_ui.py` يعرض نتائج
- [ ] `pytest tests/ -q` ينجح

---

## 13) ترتيب التشغيل الموصى به (ملخص)

```
مرة واحدة:  venv + pip install
كل فهرسة:   preprocessing → indexer CLI
كل بحث:     preprocessing + retrieval + (streamlit)
بعد فهرسة:  reload-index أو إعادة تشغيل retrieval
```
