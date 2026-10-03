# Industrial Maintenance Knowledge Assistant

A retrieval-augmented generation (RAG) system that answers PLC alarm codes, motor/drive fault, and Modbus communication error questions — grounded in official Siemens documentation, with citations to the exact source and page, and a calibrated confidence/refusal system to avoid hallucinated troubleshooting advice.

**🔗 Live demo:** [plc-maintenance-rag-yitdqdqgmw7bmiwauv4sgb.streamlit.app]

---

## Why this project

Industrial technicians troubleshooting equipment faults have to manually search hundreds of pages of PLC manuals, alarm references, and drive documentation. This project builds an assistant that retrieves the exact relevant passage for a fault/symptom and generates a grounded, cited, step-by-step answer — with an explicit "I don't have enough evidence" refusal when the knowledge base genuinely doesn't cover the question, instead of guessing.

## Corpus

Built from three official, publicly available Siemens manuals:

| Source | Content | Records |
|---|---|---|
| `G120.pdf` | SINAMICS G120 drive — alarm/fault code table | 92 |
| `s71500_cm_ptp_function_manual.pdf` | S7-1500 PtP module — Modbus/communication error codes | 160 |
| `s71200_easy_book.pdf` | S7-1200 — diagnostic concepts (buffer, LED states) | 3 |
| **Total** | | **255** |

**Extraction challenge:** PDF tables with merged cells (two alarm codes sharing one remedy) were silently scrambled by naive text extraction (`pdftotext`). Diagnosed the root cause (table structure is lost when flattened to plain text) and switched to `pdfplumber`'s native table-geometry reader, which correctly reconstructs merged cells. See `parse_alarms_final.py`.

## Architecture

```
Query
  ↓
┌─────────────────────────────────────────┐
│ Exact-code fast path (e.g. "F01000")      │ → direct lookup, confidence 1.0
└─────────────────────────────────────────┘
  ↓ (no exact match)
┌──────────────┬──────────────┐
│ BM25 (keyword)│ Dense (fastembed)│
└──────────────┴──────────────┘
  ↓ Reciprocal Rank Fusion
┌─────────────────────────────────────────┐
│ Cross-encoder reranking (fastembed)       │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ Two-layer confidence check:               │
│  Layer 1: score < 0.15 → auto-refuse      │
│  Layer 2: 0.15–0.30 → LLM sufficiency check│
└─────────────────────────────────────────┘
  ↓
Citation-grounded answer generation (Groq API)
```

## Evaluation

17-question hand-verified ground-truth set (4 exact-code, 10 natural-language, 3 out-of-corpus for refusal testing). Every expected answer code verified to exist in the corpus before use.

| Method | Recall@1 | Recall@3 |
|---|---|---|
| BM25 only | 57.1% | 71.4% |
| Dense only | 42.9% | 57.1% |
| Hybrid (BM25+dense, RRF) | 50.0–71.4%* | 78.6% |
| **Hybrid + rerank** | **78.6%** | **85.7–100%*** |

*\*Varies slightly between the quality-optimized model set (BGE-base + BGE-reranker, used in initial Colab validation) and the memory-optimized deployment set (BGE-small + MiniLM reranker via `fastembed`) — see "Deployment journey" below. Both configurations were independently measured with `run_eval.py`.*

**Breakdown (hybrid+rerank):** exact-code 4/4 (100%), natural-language 7/10 (70%).

**Key finding:** hybrid fusion *alone* (without reranking) actually underperformed plain BM25 on Recall@1 (50.0% vs 57.1%) — RRF can be dominated by BM25's zero-score "fake ties" and dense embeddings' weak handling of near-identical alphanumeric codes. Reranking is what recovers and exceeds the baseline (→78.6%), confirming it's doing real work, not just a checkbox addition.

## Bugs found and fixed (the actual engineering work)

1. **Table extraction corruption** — merged-cell alarm/remedy pairs scrambled by `pdftotext`; fixed with `pdfplumber`'s geometric table reader.
2. **RRF fake-ties bug** — BM25's zero-score ties (arbitrary corpus-order artifacts) polluted fusion rankings, causing `F01000` to rank below `F01001` and `16#8091` to disappear from top results entirely. Fixed by filtering zero-score candidates and adding an exact-code fast path.
3. **Reranker model mismatch** — a general-purpose reranker (`ms-marco-MiniLM`, web-search-tuned) *worsened* a domain-specific query compared to no reranking; swapping to `BAAI/bge-reranker-base` and measuring both with `run_eval.py` (not just eyeballing) confirmed which was actually better.
4. **Confidence threshold overlap** — `calibrate_confidence.py` found no clean score separation between genuinely relevant and genuinely irrelevant questions using rerank score alone, motivating the two-layer (score + LLM sufficiency check) confidence design rather than a single numeric cutoff.
5. **Groq model deprecation** — `llama-3.1-8b-instant` moved to an enterprise-only tier mid-project; added a `list_available_models()` diagnostic helper and switched to `openai/gpt-oss-20b`.
6. **Reranker-swap confidence miscalibration** — switching to a lighter reranker (`Xenova/ms-marco-MiniLM-L-6-v2` via `fastembed`, for memory reasons) silently broke the calibrated 0.15/0.30 thresholds, because the new model outputs raw unbounded logits instead of the old model's ~0-1 normalized scores. A genuinely correct answer was being refused. Fixed by applying a sigmoid transform to normalize any reranker's raw score into the same comparable range — found and fixed via live production testing, not caught in offline eval.
7. **Confidence ≠ answerability** — a query topically close to the corpus (SINAMICS G120 pricing) scored "high confidence" on retrieval relevance despite the corpus not containing pricing data. The generation prompt's explicit grounding instruction (*"if sources don't fully answer, say so"*) caught this and avoided hallucination — a real example of defense-in-depth, and a documented known limitation of relevance-based confidence scoring.

## Deployment journey

Deployed to **Streamlit Community Cloud** (free, permanent). Getting there required real debugging:

- **PyTorch memory overhead:** the original stack (`sentence-transformers` + local Qwen2.5-1.5B) needed ~4.5GB RAM — ~9x over Render's free 512MB limit. Swapped local LLM generation to the **Groq API** (removes ~3GB) and swapped `sentence-transformers` (PyTorch-based) to **`fastembed`** (ONNX Runtime-based) for both embedding and reranking — PyTorch's baseline import overhead alone (~300-400MB) was a bigger problem than model weight size. This reduced total footprint by ~85% with **no loss in Recall@1** (78.6% held constant).
- **ChromaDB version mismatch:** a database built with chromadb 1.5.9 failed to load under a different installed version on the host — fixed by pinning the exact version in `requirements.txt`.
- **Secrets handling across platforms:** Render uses plain environment variables; Streamlit Cloud uses `st.secrets`. Code checks both, so the same codebase deploys to either without modification.
- Render's free tier ultimately proved too memory-constrained even after optimization (intermittent OOM right at the boundary); Streamlit Community Cloud's free tier had enough headroom for the final, optimized stack.

## Tech stack

Python · ChromaDB · `fastembed` (ONNX Runtime) · `rank_bm25` · Groq API (`openai/gpt-oss-20b`) · Streamlit · `pdfplumber`

## Repo structure

```
embed_and_store.py       # corpus embedding + ChromaDB storage
hybrid_search.py         # BM25 + dense retrieval with RRF fusion
rerank.py                 # cross-encoder reranking + sigmoid normalization
generate_answer.py        # confidence logic + citation-grounded generation
eval_set.py / run_eval.py # ground-truth eval harness (Recall@k)
calibrate_confidence.py   # data-driven confidence threshold calibration
streamlit_app.py           # deployed UI
unified_corpus.json        # 255-record normalized corpus
chroma_db/                 # persisted vector store
```
