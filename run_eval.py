"""
Runs BM25-only, dense-only, hybrid, and hybrid+rerank retrieval against
eval_set.py and reports Recall@1 and Recall@3 for each method.

Recall@k = fraction of questions where the correct answer appears in the
top-k retrieved results. This is the core metric for judging retrieval
quality - it answers "does the system actually find the right document,"
before we even get to answer generation.

RUN THIS IN COLAB - needs hybrid_search.py, rerank.py, unified_corpus.json,
and the chroma_db folder all present.

Usage:
    python run_eval.py
"""
from eval_set import EVAL_SET
from rerank import RerankingRetriever


def is_correct(record, question):
    """A retrieved record counts as correct if its code matches expected_code,
    OR (for the one conceptual question) its doc_type matches expected_doc_type."""
    if question.get("expected_code"):
        return record["code"] == question["expected_code"]
    if question.get("expected_doc_type"):
        return record["doc_type"] == question["expected_doc_type"]
    return False  # out_of_corpus questions have no correct retrievable answer


def recall_at_k(retrieved_records, question, k):
    top_k = retrieved_records[:k]
    return any(is_correct(r, question) for r in top_k)


def main():
    retriever = RerankingRetriever()

    # scorable questions only (exclude out_of_corpus - those are for refusal testing)
    scorable = [q for q in EVAL_SET if q["type"] != "out_of_corpus"]
    print(f"\nEvaluating on {len(scorable)} scorable questions "
          f"({len(EVAL_SET) - len(scorable)} out_of_corpus questions excluded - "
          f"those test refusal behavior, added in a later step)\n")

    methods = {
        "BM25 only": lambda q: [
            retriever.by_id[i] for i in retriever.bm25_search(q, top_k=3)
        ],
        "Dense only": lambda q: [
            retriever.by_id[i] for i in retriever.dense_search(q, top_k=3)
        ],
        "Hybrid (BM25+dense, RRF)": lambda q: [
            r for r, _ in retriever.hybrid_search(q, top_k=3)
        ],
        "Hybrid + rerank": lambda q: [
            r for r, _, _ in retriever.search(q, top_k=3)
        ],
    }

    results = {}
    # cache every method's retrieved results per question ONCE - avoids
    # re-running the (slow) reranker a second time for the breakdown section
    retrieved_cache = {name: {} for name in methods}

    for method_name, method_fn in methods.items():
        print(f"Running: {method_name} ...")
        r1_hits, r3_hits = 0, 0
        for q in scorable:
            retrieved = method_fn(q["query"])
            retrieved_cache[method_name][q["query"]] = retrieved
            if recall_at_k(retrieved, q, k=1):
                r1_hits += 1
            if recall_at_k(retrieved, q, k=3):
                r3_hits += 1
        results[method_name] = {
            "recall@1": r1_hits / len(scorable),
            "recall@3": r3_hits / len(scorable),
        }

    print(f"\n{'Method':<28} {'Recall@1':>10} {'Recall@3':>10}")
    print("-" * 50)
    for method_name, scores in results.items():
        print(f"{method_name:<28} {scores['recall@1']:>9.1%} {scores['recall@3']:>9.1%}")

    # breakdown by question type - reuses cached results, no recomputation
    print("\n--- Hybrid+rerank breakdown by question type ---")
    by_type = {}
    for q in scorable:
        retrieved = retrieved_cache["Hybrid + rerank"][q["query"]]
        hit = recall_at_k(retrieved, q, k=1)
        by_type.setdefault(q["type"], []).append(hit)
    for qtype, hits in by_type.items():
        print(f"  {qtype}: {sum(hits)}/{len(hits)} correct @1")


if __name__ == "__main__":
    main()
