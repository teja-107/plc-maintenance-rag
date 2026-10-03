"""
Calibrates a rerank-score threshold for the refusal system ("I don't have
enough evidence to answer this").

Logic: run the full hybrid+rerank pipeline on BOTH the scorable questions
(which have a real answer in the corpus) and the out_of_corpus questions
(which have NO real answer). If there's a score gap between these two
groups, that gap becomes a defensible, data-driven threshold - rather than
a guessed number.

RUN THIS IN COLAB - needs the same setup as run_eval.py.

Usage:
    python calibrate_confidence.py
"""
from eval_set import EVAL_SET
from rerank import RerankingRetriever


def main():
    retriever = RerankingRetriever()

    in_corpus = [q for q in EVAL_SET if q["type"] in ("natural", "exact_code")]
    out_of_corpus = [q for q in EVAL_SET if q["type"] == "out_of_corpus"]

    print("=== In-corpus questions (should have HIGH confidence) ===")
    in_corpus_scores = []
    for q in in_corpus:
        results = retriever.search(q["query"], top_k=1)
        record, fused_score, rerank_score = results[0]
        score_display = "1.0 (exact match)" if rerank_score is None else f"{rerank_score:.3f}"
        print(f"  {score_display:>20}  {q['query'][:55]}")
        if rerank_score is not None:  # exclude exact-match fast-path from stats
            in_corpus_scores.append(rerank_score)

    print("\n=== Out-of-corpus questions (should have LOW confidence) ===")
    out_of_corpus_scores = []
    for q in out_of_corpus:
        results = retriever.search(q["query"], top_k=1)
        record, fused_score, rerank_score = results[0]
        print(f"  {rerank_score:>20.3f}  {q['query'][:55]}")
        out_of_corpus_scores.append(rerank_score)

    print("\n=== Summary ===")
    if in_corpus_scores:
        print(f"In-corpus rerank scores:      min={min(in_corpus_scores):.3f}  "
              f"max={max(in_corpus_scores):.3f}  "
              f"avg={sum(in_corpus_scores)/len(in_corpus_scores):.3f}")
    if out_of_corpus_scores:
        print(f"Out-of-corpus rerank scores:   min={min(out_of_corpus_scores):.3f}  "
              f"max={max(out_of_corpus_scores):.3f}  "
              f"avg={sum(out_of_corpus_scores)/len(out_of_corpus_scores):.3f}")

    if in_corpus_scores and out_of_corpus_scores:
        gap_low = min(in_corpus_scores)
        gap_high = max(out_of_corpus_scores)
        if gap_low > gap_high:
            suggested = (gap_low + gap_high) / 2
            print(f"\nClean separation found. Suggested refusal threshold: {suggested:.3f}")
            print(f"(scores below this -> refuse / 'insufficient evidence')")
        else:
            print(f"\nNo clean separation - in-corpus min ({gap_low:.3f}) is BELOW "
                  f"out-of-corpus max ({gap_high:.3f}).")
            print("This means some in-corpus and out-of-corpus scores overlap - "
                  "a single threshold will misclassify some cases either way. "
                  "Report this honestly rather than picking an arbitrary cutoff.")


if __name__ == "__main__":
    main()
