# Project Notes - تقييم نقدي وجاهزية Task 5

## Missing Features أو أجزاء غير مكتملة
- نص assignment الأصلي يطلب two datasets، لكن لديكم supervisor-approved exception باستخدام one dataset؛ يجب توثيق هذا الاستثناء صراحةً في التقرير النهائي.
- متطلبات evaluation الأساسية (MAP, Recall, P@10, nDCG قبل/بعد الميزات الإضافية) غير منفذة بشكل مكتمل ضمن المسار الحالي.
- `evaluation_service` و`api_gateway` موجودان كبنية لكن التنفيذ الفعلي محدود.
- UI لا تعرض اختيار dataset بشكل واضح.

## Technical Debt
- وجود مسارين للفهارس (`index_data` و`indexing_service/index_data`) يسبب ambiguity.
- hardcoded URLs/ports للخدمات.
- غياب dependency manifests مكتملة ومنظمة.
- عدم وجود integration tests لعقود التواصل بين services.

## Potential Bugs / Design Issues
- VSM في `retrieval_service` لا يطبق full cosine normalization.
- `search_parallel` يستخدم `list.index()` بشكل متكرر (تعقيد O(n^2) غير ضروري).
- `top_n_filter=2` في serial hybrid قد يضعف recall.
- embedding retrieval يعتمد full scan على كل docs (ضعيف scalability).
- سلوك preprocessing قد يختلف حسب توفر spaCy model.

## Deviations and Clarifications
- متطلب الحجم الكبير للـ dataset غير ظاهر في artifacts الحالية (الموجود حالياً صغير/تجريبي).
- استخدام one dataset **ليس deviation** لأنه supervisor-approved، لكن يجب ذكره رسمياً ضمن deliverables.
- formal IR evaluation pipeline ما زالت غائبة.
- بنية SOA بدأت بشكل جيد لكنها ليست مكتملة لكل الخدمات المطلوبة.

## Scalability / Performance / Maintainability Risks
- JSON-based storage غير مناسب للأحجام الكبيرة جداً.
- إعادة تهيئة retrieval engines بكل query تزيد latency.
- لا يوجد configuration management قياسي (env-based settings).
- غياب index manifest/versioning يضعف reproducibility.

## فحص منهجي لكل Task (1-4)

### Task 1 (Preprocessing)
- Dataset size مناسب؟ **جزئياً**.
- preprocessing/preparation صحيحة؟ **غالباً نعم**.
- algorithms/evaluation مناسبة؟ **algorithms نعم**، evaluation impact غير مقاس.
- metrics meaningful؟ **غير مطبق مباشرة**.
- leakage/overfitting risks؟ **منخفضة**.
- التوافق مع IR best practices؟ **جيد كبداية**.

### Task 2 (Representation)
- Dataset size مناسب؟ **نظرياً نعم**، عملياً التنفيذ الحالي غير مقاس على حجم كبير.
- split/preparation صحيحة؟ **ناقصة توثيقاً**.
- algorithms/evaluation مناسبة؟ **نعم baseline قوي**.
- metrics meaningful؟ **غير مطبقة بعد**.
- flaws/leakage/overfitting؟ **لا leakage واضح**، لكن هناك VSM normalization gap وserial candidate gap.
- IR best practices؟ **جزئي**.

### Task 3 (Indexing)
- Dataset size مناسب؟ **ضعيف مع JSON-only approach**.
- split/preparation صحيحة؟ **غير كافية الأدلة**.
- algorithms/evaluation مناسبة؟ **نعم كبنية أساسية**.
- metrics meaningful؟ **لا توجد indexing QA metrics واضحة**.
- flaws/leakage/overfitting؟ **لا leakage واضح** لكن توجد مخاطر incomplete indexing.
- IR best practices؟ **متوسطة**.

### Task 4 (Query Processing)
- Dataset size مناسب؟ **جزئياً**.
- split/preparation صحيحة؟ **معقولة من ناحية preprocessing consistency**.
- algorithms/evaluation مناسبة؟ **نعم baseline جيد**.
- metrics meaningful؟ **غير مدمجة**.
- flaws/leakage/overfitting؟ **flaws موجودة (VSM/latency/serial depth)**.
- IR best practices؟ **متوسطة**.

## Task 5 Readiness Assessment

### ما الذي يعمل جيداً
- فصل واضح للخدمات الأساسية (preprocessing/indexing/retrieval).
- وجود عدة retrieval paradigms (VSM, BM25, Embedding, Hybrid).
- دعم dynamic BM25 params في API وUI.
- preprocessing consistency بين docs وqueries جيدة نسبياً.

### ما الذي يجب إصلاحه قبل المتابعة
- توثيق dataset scope المعتمد (one dataset) بشكل رسمي + التحقق من scale/qrels.
- بناء evaluation pipeline مكتمل (MAP, Recall, P@10, nDCG).
- توحيد index storage path.
- تصحيح VSM cosine normalization وتوسيع serial reranking candidate pool.

### Assumptions / Uncertainties
- التقييم مبني على الملفات الحالية الظاهرة فقط.
- artifacts الموجودة تبدو تجريبية وقد لا تمثل آخر full run.
- لم يتم تنفيذ runtime validation شامل في هذه المراجعة.

### الحكم النهائي على الجاهزية
- **الوضع الحالي: جاهزية جزئية (Partially Ready).**
- يمكن بدء Task 5 إذا كانت exploratory، لكن للوصول إلى مستوى تسليم قوي أكاديمياً يجب إغلاق النقاط السابقة أولاً.
