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
from fastembed.rerank.cross_encoder import TextCrossEncoder
from hybrid_search import HybridRetriever, reciprocal_rank_fusion

# MEMORY-CONSTRAINED DEPLOYMENT NOTE: swapped from sentence-transformers'
# CrossEncoder (PyTorch-based) to fastembed's TextCrossEncoder (ONNX
# Runtime-based) for the same reason as the embedding model swap in
# hybrid_search.py - PyTorch's baseline memory overhead alone exceeded
# Render's 512MB free tier, even with small model weights. This ONNX-based
# reranker uses the same underlying MiniLM architecture, just without the
# PyTorch runtime overhead.
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")


DISABLE_RERANK = os.environ.get("DISABLE_RERANK", "false").lower() == "true"
# Fallback for severe memory constraints: skip loading the reranker model
# entirely, falling back to hybrid (BM25+dense) retrieval only. Measured
# cost of this fallback (from run_eval.py, with the small MiniLM embedding
# model): Recall@1 71.4%, Recall@3 78.6% (vs 78.6%/85.7% with reranking) -
# a real, documented quality tradeoff, not a guess.


class RerankingRetriever(HybridRetriever):
    def __init__(self):
        super().__init__()
        if DISABLE_RERANK:
            print("DISABLE_RERANK=true - skipping reranker model entirely "
                  "(memory-constrained fallback, uses hybrid-only retrieval)")
            self.reranker = None
        else:
            print(f"Loading reranker model: {RERANKER_MODEL} ...")
            self.reranker = TextCrossEncoder(model_name=RERANKER_MODEL)

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

        if self.reranker is None:
            # DISABLE_RERANK mode: the 0.15/0.30 confidence thresholds in
            # generate_answer.py were calibrated on cross-encoder rerank
            # scores (0-1 range) and do NOT transfer to RRF fused scores
            # (a different, much smaller scale) - rather than silently
            # produce meaningless confidence labels, return a sentinel
            # (-1.0) so generate_answer.py can detect this mode explicitly
            # and skip threshold-based confidence, instead of guessing.
            top = candidates[:top_k]
            return [(record, fused_score, -1.0) for record, fused_score in top]

        documents = [record["text"] for record, _ in candidates]
        rerank_scores = list(self.reranker.rerank(query, documents))

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
