"""
Adds a cross-encoder reranking stage on top of hybrid_search.py's fused
results. A cross-encoder reads the (query, candidate) pair together in one
forward pass, so it can judge relevance far more precisely than comparing
separate embeddings by distance - at the cost of being much slower, which is
why we only run it on the top ~15 fused candidates, not the whole corpus.

RUN THIS IN COLAB - needs the embedding model + chroma_db from before, plus
a new cross-encoder model (downloads on first run, ~90MB).

Usage:
    python rerank.py
"""
import os
from sentence_transformers import CrossEncoder
from hybrid_search import HybridRetriever, reciprocal_rank_fusion

# MEMORY-CONSTRAINED DEPLOYMENT NOTE: BAAI/bge-reranker-base (1.1GB) gave the
# best measured results (Recall@1 78.6%, Recall@3 100% in run_eval.py) but is
# too large for Render's 512MB free tier alongside the embedding model.
# Swapped to the much smaller ms-marco-MiniLM-L-6-v2 (~90MB) for deployment.
# KNOWN LIMITATION: this smaller model was only spot-checked on a few queries
# earlier (where it mis-ranked the "motor overload" case), not run through
# the full eval harness - re-running run_eval.py with this model would give
# real Recall@k numbers instead of relying on the earlier spot-check finding.
# Override via env var to use the larger model where memory allows (Colab).
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


class RerankingRetriever(HybridRetriever):
    def __init__(self):
        super().__init__()
        print(f"Loading reranker model: {RERANKER_MODEL} ...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def search(self, query, top_k=3, candidate_k=15):
        # exact-code fast path still bypasses everything, same as before
        exact = self.exact_code_lookup(query)
        if exact:
            return [(exact, 1.0, None)]  # (record, fused_score, rerank_score)

        bm25_ids = self.bm25_search(query, top_k=candidate_k)
        dense_ids = self.dense_search(query, top_k=candidate_k)
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids])

        # take a generous pool of fused candidates, then rerank them
        pool = fused[:candidate_k]
        candidates = [(self.by_id[doc_id], fused_score) for doc_id, fused_score in pool]

        pairs = [(query, record["text"]) for record, _ in candidates]
        rerank_scores = self.reranker.predict(pairs)

        combined = [
            (record, fused_score, float(rerank_score))
            for (record, fused_score), rerank_score in zip(candidates, rerank_scores)
        ]
        combined.sort(key=lambda x: x[2], reverse=True)  # sort by rerank score
        return combined[:top_k]


def main():
    retriever = RerankingRetriever()

    test_queries = [
        "motor overload alarm what should I check",
        "PLC keeps losing connection to a Modbus device",
        "F01000",
        "diagnostic buffer how many events",
    ]

    for query in test_queries:
        print(f"\n{'='*70}\nQuery: {query!r}")
        results = retriever.search(query, top_k=3)
        for rank, (record, fused_score, rerank_score) in enumerate(results, 1):
            code = record["code"] or record["doc_type"]
            score_str = "exact match" if rerank_score is None else f"rerank={rerank_score:.3f}"
            print(f"\n  #{rank} ({score_str}) [{code}]")
            print(f"     {record['text'][:150]}")


if __name__ == "__main__":
    main()
