"""Day 3a: An in-memory vector database.

Interview talking points:
- The vector DB interface every product exposes: upsert (id, vector,
  metadata), query by vector + top-k, optional structured metadata filter.
- Exact (flat) search is O(N*d): compare the query with every vector. This
  is what we implement. Production ANN indexes (HNSW graphs, IVF clusters)
  trade a little recall for big latency wins; the API above stays the same.
- Pre-filtering (filter first, then search) vs post-filtering (search,
  then drop) -- here we pre-filter, the safer default for small filters.

Run:  uv run src/generated/day3_rag/vector_store.py
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from src.generated.common.corpus import DOCUMENTS, QA_PAIRS
from src.generated.day2_embeddings.embeddings import (
    HashEmbedder,
    chunk_corpus,
    cosine_dense,
)


@dataclass
class Entry:
    id: str
    vector: list[float]
    metadata: dict = field(default_factory=dict)
    text: str = ""


@dataclass
class Hit:
    id: str
    score: float
    text: str
    metadata: dict


class VectorStore:
    """Flat, exact-search vector store. The ANN story: replace `_search_all`
    with an HNSW/IVF index and nothing else in this class changes."""

    def __init__(self) -> None:
        self._entries: dict[str, Entry] = {}

    def upsert(
        self, id: str, vector: list[float], metadata: dict | None = None, text: str = ""
    ) -> None:
        self._entries[id] = Entry(
            id=id, vector=vector, metadata=metadata or {}, text=text
        )

    def delete(self, id: str) -> None:
        self._entries.pop(id, None)

    def count(self) -> int:
        return len(self._entries)

    def search(
        self, query_vector: list[float], k: int = 5, where: dict | None = None
    ) -> list[Hit]:
        """Pre-filter: only entries matching ALL `where` keys are scored."""
        qnorm = math.sqrt(sum(v * v for v in query_vector)) or 1.0
        q = [v / qnorm for v in query_vector]
        scored: list[tuple[float, Entry]] = []
        for e in self._entries.values():
            if where and not all(
                e.metadata.get(key) == val for key, val in where.items()
            ):
                continue
            scored.append((cosine_dense(q, e.vector), e))
        scored.sort(key=lambda t: (-t[0], t[1].id))
        return [
            Hit(id=e.id, score=s, text=e.text, metadata=e.metadata)
            for s, e in scored[:k]
        ]

    # persistence: a real store has a WAL; JSON shows the concept cheaply
    def save(self, path: str) -> None:
        data = {
            e.id: {"vector": e.vector, "metadata": e.metadata, "text": e.text}
            for e in self._entries.values()
        }
        Path(path).write_text(json.dumps(data))

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        for id, rec in data.items():
            self.upsert(id, rec["vector"], rec["metadata"], rec["text"])


def build_store(dim: int = 64) -> VectorStore:
    """Index the corpus chunks with the toy dense embedder."""
    store = VectorStore()
    embedder = HashEmbedder(dim=dim)
    for i, chunk in enumerate(chunk_corpus(60, 12)):
        store.upsert(f"c{i}", embedder.embed(chunk.text), chunk.metadata, chunk.text)
    return store


def demo() -> None:
    store = build_store()
    embedder = HashEmbedder(dim=64)
    print(f"== Vector store: {store.count()} chunks indexed (flat, exact) ==")

    qa = QA_PAIRS[1]  # "How many chunks does a typical RAG pipeline retrieve..."
    hits = store.search(embedder.embed(qa["question"]), k=3)
    print(f"\nquery: {qa['question']}")
    for h in hits:
        print(f"  {h.score:+.3f}  [{h.metadata['doc_id']}] {h.text[:60]}...")

    print("\n== Metadata filtering (pre-filter, then similarity) ==")
    hits = store.search(
        embedder.embed("retrieval quality"), k=3, where={"section": "retrieval"}
    )
    for h in hits:
        print(
            f"  {h.score:+.3f}  section={h.metadata['section']:<10} doc={h.metadata['doc_id']}"
        )
    # filter guarantees: every hit comes from the requested section
    assert all(h.metadata["section"] == "retrieval" for h in hits)

    print("\n== Persistence ==")
    path = "/tmp/pi_vecstore.json"
    store.save(path)
    store2 = VectorStore()
    store2.load(path)
    qv = embedder.embed("hybrid search")
    assert [(h.id, round(h.score, 6)) for h in store.search(qv, 5)] == [
        (h.id, round(h.score, 6)) for h in store2.search(qv, 5)
    ]
    print(f"  saved {store.count()} vectors to {path}, reloaded -> identical top-5")

    print("\n== Cost of exact search ==")
    print(f"  every query compares against all {store.count()} vectors; ANN indexes")
    print("  (HNSW/IVF) cut that to a few hundred probes at ~95-99% recall")


if __name__ == "__main__":
    demo()
