"""
BM25 keyword-based retrieval over the unified corpus. This is the "sparse"
half of your hybrid retrieval system - no embedding model needed, so it runs
anywhere (unlike the dense/semantic half in embed_and_store.py).

Exact-match strength: BM25 is much better than dense embeddings at finding
exact codes like "F01000" or "16#8091" - dense embeddings sometimes blur
these together since they weren't a strong signal in training data.

Usage:
    python bm25_search.py
"""
import json
import re
from rank_bm25 import BM25Okapi

CORPUS_PATH = "unified_corpus.json"


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9#]+", text.lower())


def main():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        records = json.load(f)

    corpus_texts = [r["text"] for r in records]
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    test_queries = [
        "motor overload alarm what should I check",
        "F01000",
        "16#8091",
        "diagnostic buffer how many events",
        "Modbus slave address error",
    ]

    for query in test_queries:
        print(f"\n{'='*70}\nQuery: {query!r}")
        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

        for rank, idx in enumerate(top_idx, 1):
            r = records[idx]
            print(f"\n  #{rank} (score={scores[idx]:.2f}) [{r['code'] or r['doc_type']}]")
            print(f"     {r['text'][:140]}")


if __name__ == "__main__":
    main()
