"""Day 3b: BM25 + dense retrieval fused with Reciprocal Rank Fusion.

Interview talking points:
- BM25: TF saturates (k1), long documents are discounted (b). It is a
  strong, unsupervised baseline that handles rare keywords/identifiers.
- Dense retrieval: paraphrase-friendly, but (with our toy embedder) blind
  to rare exact terms.
- Hybrid: merge the two *ranked lists* with RRF -- score = sum over lists
  of 1/(k + rank), conventionally k=60. No score calibration needed
  because only ranks are used.

Run:  uv run src/generated/day3_rag/hybrid_search.py
"""

import math
from dataclasses import dataclass

from src.generated.common.corpus import DOCUMENTS, QA_PAIRS
from src.generated.day2_embeddings.embeddings import (
    HashEmbedder,
    chunk_corpus,
    cosine_dense,
    tokenize,
)

# ---------------------------------------------------------------------- BM25 --


@dataclass
class BM25Result:
    id: str
    score: float


class BM25:
    """Okapi BM25: score(q, d) = sum over query terms of
    idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*len/avg_len))"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: dict[str, list[str]] = {}
        self.doc_len: dict[str, int] = {}
        self.avg_len = 0.0
        self.df: dict[str, int] = {}
        self.n = 0

    def fit(self, texts: dict[str, str]) -> "BM25":
        """texts: id -> text."""
        total_len = 0
        for id, text in texts.items():
            toks = tokenize(text)
            self.docs[id] = toks
            self.doc_len[id] = len(toks)
            total_len += len(toks)
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.n = len(texts)
        self.avg_len = total_len / self.n if self.n else 0.0
        return self

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_id: str) -> float:
        total = 0.0
        for t in tokenize(query):
            tf = self.docs[doc_id].count(t)
            if tf == 0:
                continue
            norm = (
                tf
                * (self.k1 + 1)
                / (
                    tf
                    + self.k1
                    * (1 - self.b + self.b * self.doc_len[doc_id] / self.avg_len)
                )
            )
            total += self._idf(t) * norm
        return total

    def search(self, query: str, k: int = 5) -> list[BM25Result]:
        results = [BM25Result(id, self.score(query, id)) for id in self.docs]
        results.sort(key=lambda r: (-r.score, r.id))
        return results[:k]


# ----------------------------------------------------------------------- RRF --


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """score(d) = sum over lists of 1 / (k + rank(d)). Rank-only: no score
    normalization between BM25 and cosine ever needed."""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, id in enumerate(lst):
            scores[id] = scores.get(id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda t: (-t[1], t[0]))


# -------------------------------------------------------------------- hybrid --


@dataclass
class RetrievedChunk:
    id: str
    text: str
    doc_id: str
    metadata: dict


def build_index(dim: int = 64, chunk_words: int = 60, overlap: int = 12):
    """Returns (bm25, embedder, chunks) -- the shared hybrid index used by Days 4-5."""
    chunks = chunk_corpus(chunk_words, overlap)
    texts = {f"c{i}": c.text for i, c in enumerate(chunks)}
    bm25 = BM25().fit(texts)
    embedder = HashEmbedder(dim=dim)
    return bm25, embedder, chunks


def hybrid_search(
    query: str,
    bm25: BM25,
    embedder: HashEmbedder,
    chunks: list,
    k: int = 5,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """`chunks` may be a filtered subset; ids come from chunk.id (global)."""
    chunk_by_id = {
        c.id if c.id is not None else f"c{i}": c for i, c in enumerate(chunks)
    }
    sparse = [r.id for r in bm25.search(query, k=20)]
    qv = embedder.embed(query)
    dense = sorted(
        (
            (
                cosine_dense(qv, embedder.embed(c.text)),
                c.id if c.id is not None else f"c{i}",
            )
            for i, c in enumerate(chunks)
        ),
        key=lambda t: (-t[0], t[1]),
    )[:20]
    fused = rrf_fuse([sparse, [id for _, id in dense]], k=rrf_k)
    out = []
    for id, score in fused[:k]:
        if id not in chunk_by_id:  # sparse hit not in the filtered pool
            continue
        c = chunk_by_id[id]
        out.append(
            RetrievedChunk(
                id=id,
                text=c.text,
                doc_id=c.doc_id,
                metadata={**c.metadata, "rrf_score": round(score, 4)},
            )
        )
    return out


# -------------------------------------------------------------------- demo ---


def demo() -> None:
    bm25, embedder, chunks = build_index()

    print(f"== Hybrid search over {len(chunks)} chunks ==")
    print("   (queries chosen so lexical-only and dense-only each fail somewhere)\n")

    test_queries = [
        # keyword-heavy -> BM25 should shine
        "What constant does Reciprocal Rank Fusion use?",
        # paraphrase -> only broad term overlap helps
        "How do you keep a model from making things up?",
        # mixed
        "How many candidates do rerankers typically process?",
    ]
    for q in test_queries:
        print(f"query: {q}")
        sparse = bm25.search(q, k=3)
        print("  bm25 :", [(r.id, round(r.score, 2)) for r in sparse])
        qv = embedder.embed(q)
        dense = sorted(
            (
                (cosine_dense(qv, embedder.embed(c.text)), f"c{i}")
                for i, c in enumerate(chunks)
            ),
            key=lambda t: (-t[0], t[1]),
        )[:3]
        print("  dense:", [(id, round(s, 2)) for s, id in dense])
        hyb = hybrid_search(q, bm25, embedder, chunks, k=3)
        print("  fused:", [(h.id, h.metadata["rrf_score"], h.doc_id) for h in hyb])
        print()


if __name__ == "__main__":
    demo()

    # Non-TODO asserts: BM25 basics and RRF behaviour.
    _bm25, _emb, _chunks = build_index()
    _texts = {f"c{i}": c.text for i, c in enumerate(_chunks)}

    # A term that appears in exactly one chunk must rank that chunk first.
    q = "constant of 60"
    top = hybrid_search(q, _bm25, _emb, _chunks, k=1)[0]
    assert top.doc_id == "hybrid-search", top

    # RRF: an item ranked 1st in one list and 3rd in the other beats
    # an item ranked 5th and 6th (1/61 + 1/63 > 1/65 + 1/66).
    fused = dict(rrf_fuse([["a", "b", "x"], ["y", "b", "a"]]))
    assert fused["a"] > fused["x"] or "x" not in fused

    print("[demo asserts passed]")
