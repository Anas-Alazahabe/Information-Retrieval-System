# Task 01 - Data Pre-Processing

## Requirement Description
مطلوب في Task 1 تنفيذ preprocessing للنصوص قبل indexing والاسترجاع باستخدام تقنيات مناسبة مثل normalization, stopword removal, stemming, lemmatization.

## ما الذي تم تنفيذه
- إنشاء microservice مستقلة باستخدام FastAPI داخل `preprocessing_service`.
- Endpoints المتوفرة:
  - `/preprocess` لمعالجة نص/استعلام واحد.
  - `/preprocess-batch` لمعالجة Batch من النصوص.
- pipeline التنظيف في `preprocessing_service/app/core/cleaner.py` تشمل:
  - حذف الروابط.
  - lowercase + whitespace normalization.
  - مسارين:
    - lemmatization mode عبر spaCy (إذا model متوفر).
    - traditional mode عبر regex/punctuation cleanup ثم split.
  - stopword removal (NLTK English stopwords).
  - optional stemming (Porter).

## الملفات والمكونات المرتبطة
- `preprocessing_service/app/main.py`
- `preprocessing_service/app/core/cleaner.py`
- التكامل مع:
  - `indexing_service/app/core/indexer.py`
  - `retrieval_service/app/main.py`

## Algorithms and Techniques
- Rule-based normalization/token filtering.
- NLTK stopword filtering.
- Porter stemming.
- spaCy lemmatization مع fallback behavior.

## Inputs and Outputs
- Input: raw text(s) + flags (`use_stemming`, `use_lemmatization`, `remove_stopwords`).
- Output: token lists + counts.

## تقييم جودة التنفيذ من منظور IR
- **هل dataset size مناسب؟** مبدئياً نعم كمنهجية preprocessing، لكن artifacts الحالية صغيرة جداً للتأكد النهائي.
- **هل preprocessing/preparation صحيحة؟** غالباً نعم لأن نفس الخدمة تُستخدم مع docs وquery.
- **هل algorithm مناسبة؟** مناسبة كبداية قوية للـ lexical IR.
- **هل metrics محسوبة؟** لا يوجد قياس مباشر لأثر preprocessing حالياً.
- **مخاطر منهجية؟**
  - المعالجة حالياً English-focused.
  - سلوك spaCy يعتمد على توفر model في البيئة (قد يسبب اختلاف النتائج).
  - عدم وجود logging واضح لإعدادات preprocessing المستخدمة في كل run.
- **التوافق مع IR best practices؟** جيد كبداية، لكنه يحتاج reproducibility controls.

## Observations and Recommendations
- حفظ preprocessing configuration داخل metadata (language/options/model versions).
- إضافة deterministic tests لحالات edge مثل empty/URL/numeric/noisy text.
- تنفيذ token audit samples على dataset قبل full indexing.
- توثيق preprocessing parity بين index-time وquery-time ضمن evaluation report.
