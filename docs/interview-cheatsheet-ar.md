# دليل المقابلة — نظام استرجاع المعلومات 2026

## جملة افتتاحية (30 ثانية)

> نظام SOA بسبع خدمات FastAPI: معالجة، استرجاع، تحسين استعلام، تخصيص، تجميع، RAG، وتقييم. الفهرس: 200K مقطع MS MARCO؛ التقييم على dev qrels فقط.

## الخدمات — ماذا تقول لكل واحدة؟

| الخدمة | المنفذ | مسؤوليتها | ملف الدخول |
|--------|--------|-----------|------------|
| preprocessing | 8000 | توكنة، lemmatization، stopwords | `preprocessing_service/app/main.py` |
| retrieval | 8002 | مطابقة وترتيب (VSM/BM25/Emb/Hybrid) | `retrieval_service/app/main.py` |
| query_refinement | 8003 | PRF، مرادفات، preprocess | `query_refinement_service/app/main.py` |
| personalization | 8004 | ملف اهتمام + إعادة ترتيب | `personalization_service/app/main.py` |
| clustering | 8005 | t-SNE وتجميع (عرض فقط) | `clustering_service/app/main.py` |
| rag | 8006 | إجابة Gemini من المقاطع | `rag_service/app/main.py` |
| indexing | CLI | بناء `index_data/` | `indexing_service/app/core/indexer.py` |
| evaluation | CLI/API | MAP, nDCG على dev qrels | `evaluation_service/app/main.py` |
| UI | 8501 | Streamlit + `shared/search_pipeline.py` | `app_ui.py` |

## الميزات الإضافية — لماذا؟

| الميزة | الفائدة | دليل رقمي |
|--------|---------|-----------|
| FAISS | بحث دلالي سريع على 200K متجه | Embedding vs BM25 ΔnDCG@10 في `FINAL_EVAL_SUMMARY.md` |
| Hybrid RRF | دمج لفظي + دلالي | chart `04_vector_store_comparison.png` |
| Query refinement | أسئلة طبيعية ومرادفات | ΔnDCG في تقرير refinement |
| Personalization | ترتيب حسب اهتمام المستخدم | أوزان alpha من الواجهة |
| RAG | إجابة نصية مع استشهاد | citation_rate في eval RAG |
| Clustering | فهم توزيع المجموعة | t-SNE في الواجهة (لا يغيّر الترتيب) |

## أسئلة متوقعة

**لماذا 200K وليس 8.8M؟**  
ذاكرة ووقت عرض؛ موثّق في الواجهة و`docs/dataset-scope-ar.md`. استثناء مجموعة واحدة بموافقة المشرف.

**كيف يعمل Hybrid؟**  
`docs/hybrid-explained-ar.md` — RRF أو serial rerank.

**أين تُخزَّن النماذج؟**  
فهارس JSON مضغوطة (`.json.gz`) + FAISS + تحميل عند بدء retrieval (لا تدريب عند أول استعلام).

**كيف تُعرض الوثائق؟**  
MySQL `documents` — معرّف كامل + نص أصلي كامل في الواجهة.

## تشغيل سريع

```powershell
.\scripts\start_stack.ps1
streamlit run app_ui.py
```
