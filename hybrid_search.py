"""
Hybrid retrieval: combines BM25 (keyword) + dense (semantic embedding) search
using Reciprocal Rank Fusion (RRF).

Why hybrid: our sanity check showed dense search alone missed the exact
"motor overload" alarm codes (A07805, F30005) and instead surfaced related-
but-wrong codes (overtemperature, maintenance interval). BM25 found the
correct codes immediately because "overload" is an exact term match. RRF
combines both so a query benefits from whichever method handles it better,
without needing to guess in advance which one that'll be.

RUN THIS IN COLAB (same environment as embed_and_store.py) - it needs both
the embedding model and the persisted chroma_db/ folder from that script.

Usage:
    python hybrid_search.py
"""
import os
import json
import re
import chromadb
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding

CORPUS_PATH = "unified_corpus.json"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "plc_maintenance_kb"
# Switched from sentence-transformers (PyTorch-based, ~300-400MB baseline
# overhead just from importing torch) to fastembed (ONNX Runtime-based,
# inference-only, much smaller footprint) to fit Render's free-tier 512MB
# RAM limit. MUST match the model used in embed_and_store.py to build
# chroma_db - embeddings from different models aren't compatible.
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
USE_BGE_PREFIX = "bge" in MODEL_NAME.lower()
RRF_K = 60  # standard RRF constant - dampens the influence of any single rank


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9#]+", text.lower())


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """
    rankings: list of ranked id-lists, e.g. [bm25_ids_in_rank_order, dense_ids_in_rank_order]
    Returns: list of (id, fused_score) sorted best-first.
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self):
        print("Loading corpus and building BM25 index ...")
        with open(CORPUS_PATH, encoding="utf-8") as f:
            self.records = json.load(f)
        self.by_id = {r["id"]: r for r in self.records}

        tokenized_corpus = [tokenize(r["text"]) for r in self.records]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_ids = [r["id"] for r in self.records]  # positional order matches bm25 index

        print("Loading embedding model + Chroma collection ...")
        self.model = TextEmbedding(model_name=MODEL_NAME)
        client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = client.get_collection(COLLECTION_NAME)

    def bm25_search(self, query, top_k=10):
        scores = self.bm25.get_scores(tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # FIX 1: drop zero-score "fake ties" - a score of 0 means no real
        # keyword overlap at all, so including it just pollutes the fusion
        # with arbitrary corpus-order noise instead of genuine relevance.
        return [self.bm25_ids[i] for i in ranked_idx if scores[i] > 0]

    def dense_search(self, query, top_k=10):
        q_prefixed = f"query: {query}" if USE_BGE_PREFIX else query
        q_emb = list(self.model.embed([q_prefixed]))[0]
        results = self.collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
        return results["ids"][0]

    def exact_code_lookup(self, query):
        """
        FIX 2: if the query IS literally a code (or contains one), bypass
        fuzzy retrieval entirely and return the exact record. Alphanumeric
        identifiers are exactly what embeddings are worst at - there's no
        reason to risk semantic drift when an exact match is available.
        """
        q = query.strip().upper()
        for record in self.records:
            if record["code"] and record["code"].upper() == q:
                return record
        return None

    def hybrid_search(self, query, top_k=5, candidate_k=15):
        exact = self.exact_code_lookup(query)
        if exact:
            return [(exact, 1.0)]  # score 1.0 = maximum confidence, exact match

        bm25_ids = self.bm25_search(query, top_k=candidate_k)
        dense_ids = self.dense_search(query, top_k=candidate_k)
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids])
        top = fused[:top_k]
        return [(self.by_id[doc_id], score) for doc_id, score in top]


def main():
    retriever = HybridRetriever()

    test_queries = [
        "motor overload alarm what should I check",
        "F01000",
        "16#8091",
        "diagnostic buffer how many events",
        "Modbus slave address error",
    ]

    for query in test_queries:
        print(f"\n{'='*70}\nQuery: {query!r}")
        results = retriever.hybrid_search(query, top_k=3)
        for rank, (record, score) in enumerate(results, 1):
            code = record["code"] or record["doc_type"]
            print(f"\n  #{rank} (RRF score={score:.4f}) [{code}]")
            print(f"     {record['text'][:150]}")


if __name__ == "__main__":
    main()
