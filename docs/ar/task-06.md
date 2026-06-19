# المهمة 06 — مطابقة الاستعلام وترتيب النتائج

## وصف المتطلب
بناء تابع لمطابقة تمثيل الاستعلام مع الوثائق وترتيب النتائج حسب أعلى درجات التشابه، مع اعتماد طريقة المطابقة المناسبة لكل نموذج تمثيل.

## ما تم تنفيذه
- حزمة مطابقة مخصصة: `retrieval_service/app/core/matching/`
- `MatcherRegistry` يوجّه كل `representation_mode` إلى matcher مناسب
- `QueryRepresentation` يُبنى مرة واحدة بعد المعالجة المسبقة
- `Ranker` يرتب النتائج تنازلياً مع كسر التعادل حسب `doc_id`

| النمط | طريقة المطابقة |
|-------|----------------|
| `vsm` | تشابه جيبي (Cosine) على TF-IDF |
| `bm25` | صيغة BM25 |
| `embedding` | تشابه جيبي على المتجهات |
| `hybrid_parallel` | دمج RRF |
| `hybrid_serial` | ترشيح BM25 ثم إعادة ترتيب دلالية |

- دعم أداء للتضمين: حلقة، NumPy، FAISS
- واجهة API: `matcher`, `matching_method`, `params`, أزمنة منفصلة
- `GET /matchers` لعرض الأنماط المدعومة

## الملفات الرئيسية
- `retrieval_service/app/core/matching/`
- `retrieval_service/app/main.py`
- `shared/index_builder.py` (بناء FAISS)
- `evaluation_service/`
- `tests/test_matchers.py`

## تدفق البيانات
```
استعلام خام → معالجة مسبقة → QueryRepresentation → Matcher → Ranker → نتائج مرتبة
```

## التقييم
```powershell
python -m evaluation_service.run --scale preval --max-queries 100 --output-dir evaluation_results
```

## قسم التقرير العربي (مطابقة الاستعلام وترتيب النتائج)
- جدول النمط → طريقة المطابقة (انظر الجدول أعلاه)
- مثال استجابة JSON مع `results` و `matching_method`
- تبرير: Cosine لـ VSM والتضمين لأن المتجهات موحّدة الطول؛ RRF للهجين المتوازي لدمج قوائم ترتيب مستقلة
