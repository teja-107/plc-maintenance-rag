"""
Ground-truth evaluation set for the PLC/motor maintenance RAG system.

Each question has a known-correct answer code (the alarm/error record that
SHOULD be retrieved). This lets us compute real Recall@k numbers instead of
eyeballing individual query outputs.

Question types, deliberately mixed:
  - exact_code:   literal code lookups (should be trivial - tests the fast path)
  - natural:      plain-language descriptions of a symptom (the real use case)
  - cross_doc:    questions where the right answer could plausibly come from
                   more than one document - tests whether metadata/content
                   correctly disambiguates
  - out_of_corpus: questions with NO correct answer in the corpus - tests
                   refusal behavior (built in the next step, not scored here yet)
"""

EVAL_SET = [
    # --- exact code lookups ---
    {"query": "F01000", "expected_code": "F01000", "type": "exact_code"},
    {"query": "16#8091", "expected_code": "16#8091", "type": "exact_code"},
    {"query": "A07805", "expected_code": "A07805", "type": "exact_code"},
    {"query": "16#8186", "expected_code": "16#8186", "type": "exact_code"},

    # --- natural language symptom descriptions ---
    {"query": "motor is drawing too much current and overload alarm triggered",
     "expected_code": "A07805", "type": "natural"},
    {"query": "drive shows software fault after power up",
     "expected_code": "F01000", "type": "natural"},
    {"query": "Modbus slave has an invalid address configured",
     "expected_code": "16#8186", "type": "natural"},
    {"query": "motor is running too hot, what could be wrong",
     "expected_code": "A07910", "type": "natural"},
    {"query": "converter needs maintenance interval reset",
     "expected_code": "A01590", "type": "natural"},
    {"query": "memory card failed to load on the control unit",
     "expected_code": "F01044", "type": "natural"},
    {"query": "how many diagnostic events does the CPU log keep",
     "expected_code": None, "expected_doc_type": "concept_explainer", "type": "natural"},
    {"query": "torque or speed reading is higher than expected",
     "expected_code": "A07921", "type": "natural"},
    {"query": "Modbus master got no response from the slave device in time",
     "expected_code": "16#80C9", "type": "natural"},
    {"query": "firmware needs to be upgraded due to a software fault",
     "expected_code": "F01015", "type": "natural"},

    # --- out-of-corpus (no correct answer exists) - for refusal testing later ---
    {"query": "how do I replace the timing belt on a Toyota Corolla",
     "expected_code": None, "type": "out_of_corpus"},
    {"query": "what is the price of the SINAMICS G120 drive",
     "expected_code": None, "type": "out_of_corpus"},
    {"query": "how do I configure a PROFIBUS network from scratch",
     "expected_code": None, "type": "out_of_corpus"},
]


def summarize():
    from collections import Counter
    types = Counter(q["type"] for q in EVAL_SET)
    print(f"Total questions: {len(EVAL_SET)}")
    for t, count in types.items():
        print(f"  {t}: {count}")


if __name__ == "__main__":
    summarize()
