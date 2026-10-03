"""
Normalizes g120_alarms_final.json, s71500_errors.json, and
easybook_diagnostics.json into one unified schema ready for embedding.

Unified record shape:
{
    id: unique string
    text: the actual passage to embed (human-readable, self-contained)
    doc_type: "alarm_code" | "error_code" | "concept_explainer"
    machine_type: "drive" | "plc"
    model: specific hardware model string
    code: alarm/error code if applicable, else None
    source_doc: original filename
    page: page number
}

Output: unified_corpus.json
"""
import json

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_g120(records):
    out = []
    for r in records:
        text = (
            f"{r['kind']} {r['code']} ({r['model']}): {r['cause']}. "
            f"Remedy: {r['remedy']}"
        )
        out.append({
            "id": f"g120_{r['code']}_{r['page']}",
            "text": text,
            "doc_type": "alarm_code",
            "machine_type": r["machine_type"],
            "model": r["model"],
            "code": r["code"],
            "source_doc": r["source_doc"],
            "page": r["page"],
        })
    return out


def normalize_s71500(records):
    out = []
    for r in records:
        cat = f" [{r['category']}]" if r.get("category") else ""
        text = f"Error {r['code']}{cat} ({r['model']}): {r['description']}."
        if r["solution"]:
            text += f" Solution: {r['solution']}"
        out.append({
            "id": f"s71500_{r['code']}_{r['page']}",
            "text": text,
            "doc_type": "error_code",
            "machine_type": r["machine_type"],
            "model": r["model"],
            "code": r["code"],
            "source_doc": r["source_doc"],
            "page": r["page"],
        })
    return out


def normalize_easybook(records):
    out = []
    for r in records:
        text = f"{r['title']} (Section {r['section']}, {r['model']}): {r['content']}"
        out.append({
            "id": f"easybook_{r['section']}",
            "text": text,
            "doc_type": r["doc_type"],
            "machine_type": r["machine_type"],
            "model": r["model"],
            "code": None,
            "source_doc": r["source_doc"],
            "page": r["page"],
        })
    return out


if __name__ == "__main__":
    g120 = normalize_g120(load("g120_alarms_final.json"))
    s71500 = normalize_s71500(load("s71500_errors.json"))
    easybook = normalize_easybook(load("easybook_diagnostics.json"))

    unified = g120 + s71500 + easybook

    with open("unified_corpus.json", "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2)

    print(f"Unified corpus: {len(unified)} records")
    print(f"  - G120 alarms: {len(g120)}")
    print(f"  - S7-1500 errors: {len(s71500)}")
    print(f"  - Easy Book concepts: {len(easybook)}")
    print("\nSample unified record:")
    print(json.dumps(unified[0], indent=2))
    print("\nSample from each doc_type:")
    for dtype in ["alarm_code", "error_code", "concept_explainer"]:
        sample = next(r for r in unified if r["doc_type"] == dtype)
        print(f"\n[{dtype}]")
        print(sample["text"][:200])
