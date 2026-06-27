# تجربة 8.8M في نسخة معزولة (اختياري)

لا تُعدّل مشروع العرض الأصلي. انسخ المجلد بالكامل إلى سطح المكتب ثم غيّر **فقط** في النسخة الجديدة:

## 1. نسخ المشروع

```powershell
Copy-Item -Recurse "C:\Users\Golden\Documents\ir_core_project" "$env:USERPROFILE\Desktop\ir_core_fullscale"
cd "$env:USERPROFILE\Desktop\ir_core_fullscale"
```

## 2. ملف `.env` منفصل

```env
IR_INDEX_DIR=C:\Users\Golden\Desktop\ir_core_fullscale\index_data_full
IR_INDEX_SCALE=full
IR_MAX_DOCS=8800000
IR_PREPROCESS_URL=http://127.0.0.1:8010
IR_RETRIEVAL_URL=http://127.0.0.1:8012
# ... منافذ مختلفة لكل خدمة
MYSQL_DATABASE=ir_fullscale
```

## 3. قواعد الأمان

- لا تشارك `index_data/` أو MySQL مع نسخة العرض.
- لا تشغّل فهرسة كاملة وأنت متصل بنفس `.env` للعرض.
- احتفظ بنسخة العرض على 200K للمقابلة.

## 4. فهرسة (ساعات)

```powershell
python -m indexing_service.run --scale full
python migrate_to_db.py --max-docs 8800000
```

## 5. ضغط الفهارس (بعد البناء)

```powershell
python scripts/compress_index_artifacts.py --index-dir index_data_full
```

نسخة العرض الأصلية تبقى كما هي في `ir_core_project`.
