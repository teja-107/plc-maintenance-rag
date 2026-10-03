"""
Embeds unified_corpus.json using fastembed (ONNX Runtime, not PyTorch) and
stores everything in a persistent ChromaDB collection with metadata attached.

WHY fastembed instead of sentence-transformers: sentence-transformers pulls
in full PyTorch, which reserves ~300-400MB of RAM just from being imported,
before any model is even loaded. That alone exceeds Render's free-tier
512MB limit. fastembed runs the same kind of small embedding models through
ONNX Runtime instead - built purely for inference, no autograd/training
machinery - with a dramatically smaller memory footprint. This was the fix
that got the deployed version working within the free tier's memory limit.

RUN THIS ON YOUR OWN MACHINE (or Google Colab) — it needs to download the
embedding model on first run (~130MB), which requires normal internet access.

Setup (run once):
    pip install chromadb fastembed

Usage:
    python embed_and_store.py

This creates a ./chroma_db/ folder containing your persistent vector store —
you can query it again later without re-embedding.
"""
import os
import json
import chromadb
from fastembed import TextEmbedding

CORPUS_PATH = "unified_corpus.json"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "plc_maintenance_kb"
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
USE_BGE_PREFIX = "bge" in MODEL_NAME.lower()  # BGE models want "query:"/"passage:" prefixes


def main():
    print(f"Loading corpus from {CORPUS_PATH} ...")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        records = json.load(f)
    print(f"  {len(records)} records loaded")

    print(f"Loading embedding model: {MODEL_NAME} (downloads on first run) ...")
    model = TextEmbedding(model_name=MODEL_NAME)

    print("Embedding all records ...")
    texts = [r["text"] for r in records]
    # BGE models recommend a "passage:" prefix for retrieval passages
    prefixed = [f"passage: {t}" for t in texts] if USE_BGE_PREFIX else texts
    embeddings = list(model.embed(prefixed))  # fastembed returns a generator of np arrays

    print(f"Connecting to ChromaDB at {DB_PATH} ...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # fresh collection each run - delete if it already exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print("Storing records + embeddings + metadata ...")
    ids = [r["id"] for r in records]
    metadatas = []
    for r in records:
        # Chroma metadata values must be str/int/float/bool, not None
        meta = {
            "doc_type": r["doc_type"],
            "machine_type": r["machine_type"],
            "model": r["model"],
            "code": r["code"] or "",
            "source_doc": r["source_doc"],
            "page": r["page"],
        }
        metadatas.append(meta)

    # batch add (Chroma has a max batch size; 255 records fits in one call,
    # but chunked here so this script scales if you grow the corpus later)
    BATCH = 100
    for i in range(0, len(records), BATCH):
        batch_embeddings = [e.tolist() for e in embeddings[i:i+BATCH]]
        collection.add(
            ids=ids[i:i+BATCH],
            embeddings=batch_embeddings,
            documents=texts[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
        )

    print(f"\nDone. {collection.count()} records stored in collection '{COLLECTION_NAME}'.")

    # quick sanity check query
    print("\n--- Sanity check query ---")
    query = "motor overload alarm what should I check"
    q_prefixed = f"query: {query}" if USE_BGE_PREFIX else query
    q_emb = list(model.embed([q_prefixed]))[0]
    results = collection.query(query_embeddings=[q_emb.tolist()], n_results=3)
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"\n[{meta['code'] or meta['doc_type']}] (distance={dist:.3f})")
        print(doc[:150])


if __name__ == "__main__":
    main()
