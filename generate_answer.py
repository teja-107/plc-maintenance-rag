"""
Full RAG answer generation: retrieval (hybrid+rerank) -> confidence check ->
citation-grounded generation using a free, local open-source LLM.

Two-layer confidence system (built from calibrate_confidence.py findings,
which showed rerank score alone doesn't cleanly separate good/bad matches):

  Layer 1 (fast, cheap): if rerank score is very low (< LOW_THRESHOLD),
     refuse immediately without even calling the LLM - clearly irrelevant.
  Layer 2 (for the ambiguous middle zone): ask the LLM directly whether the
     retrieved context is actually sufficient to answer the question, before
     generating a final answer. This catches cases the numeric score alone
     couldn't cleanly separate.

Uses Qwen2.5-1.5B-Instruct - free, open, no API key needed, runs on Colab.

RUN THIS IN COLAB - needs everything from before (hybrid_search.py,
rerank.py, unified_corpus.json, chroma_db/) plus this will download the
Qwen model on first run (~3GB).

Usage:
    python generate_answer.py
"""
import os
from groq import Groq
from rerank import RerankingRetriever

# Switched from a local Qwen2.5-1.5B model to Groq's free hosted API.
# Reason: the combined footprint of local embedding + reranker + LLM models
# (~4.5GB) exceeded Render's free-tier 512MB RAM limit. Using a hosted API
# for generation removes the single largest memory cost (~3GB) while keeping
# everything else (retrieval, reranking, confidence logic) unchanged.
LLM_MODEL = "openai/gpt-oss-20b"  # fast, free-tier model on Groq.
# NOTE: llama-3.1-8b-instant moved to Groq's Enterprise-only tier at some
# point after this project started - a good reminder that third-party API
# model availability can change without warning, worth a fallback check
# (see list_available_models() below) if this model is ever retired too.

# from calibrate_confidence.py: out-of-corpus max was 0.274, in-corpus min
# was 0.152 - scores below LOW_THRESHOLD are refused immediately (clearly
# irrelevant); scores between LOW_THRESHOLD and HIGH_THRESHOLD go through
# the LLM sufficiency check since the numeric score alone is ambiguous there
LOW_THRESHOLD = 0.15   # below this: auto-refuse, don't even ask the LLM
HIGH_THRESHOLD = 0.30  # above this: skip the sufficiency check, just answer


def list_available_models(api_key=None):
    """Debug helper: prints every model currently available to your Groq
    API key. Useful if a model gets deprecated/moved tiers again - run this
    instead of guessing a new model name."""
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    models = client.models.list()
    for m in models.data:
        print(m.id)


class AnswerGenerator:
    def __init__(self):
        self.retriever = RerankingRetriever()
        print(f"Connecting to Groq API (model: {LLM_MODEL}) ...")
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable not set. Get a free key at "
                "console.groq.com and set it as an environment variable "
                "(locally: export GROQ_API_KEY=... ; on Render: add it in "
                "the service's Environment settings)."
            )
        self.client = Groq(api_key=api_key)

    def _chat(self, system_prompt, user_prompt, max_new_tokens=300):
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def check_sufficiency(self, query, context_text):
        """LLM-judged sufficiency check for the ambiguous confidence zone."""
        system = (
            "You judge whether retrieved technical documentation is sufficient "
            "to answer a question. Respond with exactly one word: YES or NO."
        )
        user = (
            f"Question: {query}\n\nRetrieved context:\n{context_text}\n\n"
            f"Does this context contain enough information to directly answer "
            f"the question? Respond YES or NO only."
        )
        response = self._chat(system, user, max_new_tokens=5)
        return "YES" in response.upper()

    def generate_grounded_answer(self, query, records):
        """Generate a citation-grounded answer from retrieved records."""
        context_parts = []
        for i, r in enumerate(records, 1):
            context_parts.append(
                f"[Source {i}] {r['text']} (Document: {r['source_doc']}, Page: {r['page']})"
            )
        context_text = "\n\n".join(context_parts)

        system = (
            "You are an industrial maintenance assistant. Answer ONLY using the "
            "provided sources. Every claim must cite its source number, e.g. [Source 1]. "
            "If the sources don't fully answer the question, say so explicitly. "
            "Be concise and give clear, actionable steps."
        )
        user = f"Question: {query}\n\nSources:\n{context_text}\n\nAnswer:"
        answer = self._chat(system, user, max_new_tokens=300)
        return answer, context_text

    def answer(self, query, top_k=3):
        results = self.retriever.search(query, top_k=top_k)
        top_record, top_fused, top_rerank = results[0]

        # exact-code fast path: always maximum confidence, skip everything else
        if top_rerank is None:
            answer_text, _ = self.generate_grounded_answer(query, [top_record])
            return {
                "answer": answer_text,
                "confidence": "high (exact match)",
                "sources": [top_record],
            }

        # DISABLE_RERANK mode: calibrated thresholds don't apply (see
        # rerank.py) - generate an answer but label confidence honestly as
        # unverified, rather than pretending the threshold check still works
        if top_rerank == -1.0:
            records = [r for r, _, _ in results]
            answer_text, _ = self.generate_grounded_answer(query, records)
            return {
                "answer": answer_text,
                "confidence": "unverified (reranker disabled - no calibrated confidence available)",
                "sources": records,
            }

        # Layer 1: very low score -> refuse immediately, no LLM call needed
        if top_rerank < LOW_THRESHOLD:
            return {
                "answer": (
                    "I don't have enough evidence in the knowledge base to "
                    "answer this question confidently. This may be outside "
                    "the scope of the indexed PLC/motor documentation."
                ),
                "confidence": "low (refused - below threshold)",
                "sources": [],
            }

        records = [r for r, _, _ in results]
        context_for_check = "\n\n".join(r["text"] for r in records)

        # Layer 2: ambiguous zone -> ask the LLM if context is really sufficient
        if top_rerank < HIGH_THRESHOLD:
            sufficient = self.check_sufficiency(query, context_for_check)
            if not sufficient:
                return {
                    "answer": (
                        "I found some potentially related information, but I'm "
                        "not confident it fully answers this question. Please "
                        "verify manually or rephrase your question."
                    ),
                    "confidence": f"low (LLM sufficiency check failed, rerank={top_rerank:.3f})",
                    "sources": records,
                }

        answer_text, _ = self.generate_grounded_answer(query, records)
        confidence = "high" if top_rerank >= HIGH_THRESHOLD else "medium (passed sufficiency check)"
        return {"answer": answer_text, "confidence": confidence, "sources": records}


def main():
    gen = AnswerGenerator()

    test_queries = [
        "F01000",
        "motor is drawing too much current and overload alarm triggered",
        "how do I replace the timing belt on a Toyota Corolla",
        "what is the price of the SINAMICS G120 drive",
    ]

    for query in test_queries:
        print(f"\n{'='*70}\nQuery: {query!r}")
        result = gen.answer(query)
        print(f"Confidence: {result['confidence']}")
        print(f"Answer:\n{result['answer']}")
        if result["sources"]:
            print(f"\nSources cited: {[r['code'] or r['doc_type'] for r in result['sources']]}")


if __name__ == "__main__":
    main()
