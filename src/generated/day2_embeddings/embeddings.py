"""Day 2: Embeddings (TF-IDF + toy dense) and chunking strategies.

Interview talking points:
- TF-IDF = "poor man's embedding": term frequency * inverse document
  frequency rewards terms that are distinctive *across the corpus*.
- Cosine similarity measures angle, not magnitude -- length normalization
  matters or long documents win by sheer volume.
- A real dense embedding (bi-encoder) captures paraphrase/synonymy because it
  is trained on semantic similarity; our hash embedder does not -- it is a
  deterministic stand-in so retrieval demos run offline. Keep that honest
  distinction in mind for interviews.
- Chunking: fixed-size + overlap, recursive structure-aware splitting, and
  parent-child (small-to-big) retrieval.

Run:  uv run src/generated/day2_embeddings/embeddings.py
"""

import hashlib
import math
import re
from dataclasses import dataclass

from src.generated.common.corpus import DOCUMENTS, QA_PAIRS, Document

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "and",
    "or",
    "for",
    "with",
    "what",
    "which",
    "how",
    "why",
    "does",
    "do",
    "be",
    "been",
    "it",
    "its",
    "this",
    "that",
    "as",
    "by",
    "from",
    "at",
    "into",
    "than",
    "then",
    "so",
    "can",
    "will",
    "not",
    "but",
    "they",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in STOPWORDS]


# ------------------------------------------------------------------- TF-IDF --


def term_freq(tokens: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tokens:
        out[t] = out.get(t, 0) + 1
    return out


class TfidfEmbedder:
    """Sparse vector embedder: dict of term -> tf * idf."""

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        n = len(texts)
        df: dict[str, int] = {}
        for text in texts:
            for term in set(tokenize(text)):
                df[term] = df.get(term, 0) + 1
        # Smoothed idf: log((N+1)/(df+1)) + 1 keeps every term non-zero.
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
        return self

    def embed(self, text: str) -> dict[str, float]:
        vec: dict[str, float] = {}
        for term, count in term_freq(tokenize(text)).items():
            if term in self.idf:
                vec[term] = vec.get(term, 0.0) + count * self.idf[term]
        return vec


def cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity over sparse dict vectors."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------- toy dense ---


class HashEmbedder:
    """Fixed-dim dense stand-in for a bi-encoder.

    Each token hashes to a bucket and adds +1/-1 (hash-sign); the vector is
    L2-normalized. Two texts sharing many tokens have high cosine similarity.
    Unlike a real embedding model it has NO semantics: "automobile" and
    "car" are unrelated here. That gap is exactly what real dense
    embedding models are trained to close.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _bucket_sign(self, token: str) -> tuple[int, float]:
        h = hashlib.md5(token.encode()).digest()
        bucket = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] % 2 else -1.0
        return bucket, sign

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            bucket, sign = self._bucket_sign(token)
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def cosine_dense(a: list[float], b: list[float]) -> float:
    """Dense vectors are normalized, so this is just a dot product."""
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------- chunking ---


@dataclass
class Chunk:
    text: str
    doc_id: str
    parent_id: str | None = None
    metadata: dict | None = None
    id: str | None = None  # stable index id, e.g. "c7"


def fixed_size_chunks(text: str, size_words: int = 60, overlap: int = 12) -> list[str]:
    """Fixed-size windows with overlap. Simple, fast, cuts mid-sentence."""
    words = text.split()
    out = []
    step = max(size_words - overlap, 1)
    for start in range(0, len(words), step):
        window = words[start : start + size_words]
        out.append(" ".join(window))
        if start + size_words >= len(words):
            break
    return out


def recursive_split(
    text: str,
    max_words: int = 60,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " "),
) -> list[str]:
    """Split on the largest structural separator that fits, recursively."""
    if len(text.split()) <= max_words:
        return [text.strip()] if text.strip() else []
    # find the best separator that actually occurs
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            out: list[str] = []
            current = ""
            for part in parts:
                candidate = f"{current}{sep}{part}" if current else part
                if len(candidate.split()) <= max_words:
                    current = candidate
                else:
                    if current.strip():
                        out.append(current.strip())
                    out.extend(recursive_split(part, max_words, separators))
                    current = ""
            if current.strip():
                out.append(current.strip())
            return out
    # no separator left: hard word split
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def parent_child_chunks(
    doc: Document, parent_words: int = 60, child_words: int = 30
) -> tuple[list[Chunk], list[Chunk]]:
    """Small-to-big: index children, return parents at answer time.

    Small chunks embed precisely (fewer distractions per vector); big
    chunks give the LLM enough surrounding context. Mapping child->parent
    is the whole trick of parent-document retrieval.
    """
    parents = [
        Chunk(text=t, doc_id=doc.id, parent_id=None, metadata={"section": doc.section})
        for t in fixed_size_chunks(doc.text, parent_words, 0)
    ]
    children: list[Chunk] = []
    for i, parent in enumerate(parents):
        for t in fixed_size_chunks(parent.text, child_words, 0):
            children.append(
                Chunk(
                    text=t,
                    doc_id=doc.id,
                    parent_id=f"{doc.id}:{i}",
                    metadata={"section": doc.section},
                )
            )
    return parents, children


def chunk_corpus(size_words: int = 60, overlap: int = 12) -> list[Chunk]:
    """All documents -> overlapping fixed-size chunks (used by Days 3-5)."""
    chunks: list[Chunk] = []
    for doc in DOCUMENTS:
        for i, t in enumerate(fixed_size_chunks(doc.text, size_words, overlap)):
            chunks.append(
                Chunk(
                    text=t,
                    doc_id=doc.id,
                    parent_id=f"{doc.id}:{i}",
                    metadata={
                        "doc_id": doc.id,
                        "section": doc.section,
                        "title": doc.title,
                    },
                )
            )
    for i, chunk in enumerate(chunks):
        chunk.id = f"c{i}"
    return chunks


# -------------------------------------------------------------------- demo ---


def demo() -> None:
    docs = [doc.text for doc in DOCUMENTS]

    print("== TF-IDF: query -> nearest documents ==")
    embedder = TfidfEmbedder().fit(docs)
    for qa in QA_PAIRS[:3]:
        qv = embedder.embed(qa["question"])
        scored = [
            (cosine_sparse(qv, embedder.embed(d)), doc.id)
            for doc, d in zip(DOCUMENTS, docs)
        ]
        scored.sort(reverse=True)
        top = ", ".join(f"{did}({s:.2f})" for s, did in scored[:3])
        verdict = "HIT " if scored[0][1] == qa["doc_id"] else "MISS"
        print(f"  [{verdict}] {qa['question'][:48]:<50} -> {top}")

    print("\n== Toy dense (hash) embedder, same queries ==")
    print("  (note: the hash embedder is bag-of-words with no semantics;")
    print("   misses are expected -- this is the gap real embedding models close)")
    dense = HashEmbedder(dim=64)
    qvec = {doc.id: dense.embed(doc.text) for doc in DOCUMENTS}
    for qa in QA_PAIRS[:3]:
        qv = dense.embed(qa["question"])
        scored = sorted(
            ((cosine_dense(qv, qvec[doc.id]), doc.id) for doc in DOCUMENTS),
            reverse=True,
        )
        verdict = "HIT " if scored[0][1] == qa["doc_id"] else "MISS"
        print(
            f"  [{verdict}] {qa['question'][:48]:<50} -> {scored[0][1]}({scored[0][0]:.2f})"
        )

    print("\n== Chunking strategies on one document ==")
    doc = DOCUMENTS[0]
    fixed = fixed_size_chunks(doc.text, 30, 8)
    rec = recursive_split(doc.text, 30)
    parents, children = parent_child_chunks(doc, 60, 30)
    print(
        f"  fixed(30, overlap=8): {len(fixed)} chunks; first ends '...{' '.join(fixed[0].split()[-3:])}'"
    )
    print(f"  recursive(max 30)   : {len(rec)} chunks; boundaries land on sentences")
    print(
        f"  parent-child        : {len(parents)} parents, {len(children)} children (index child, return parent)"
    )
    print(f"  rec[0][:70]         : {rec[0][:70]}...")
    print(f"  fixed[0][:70]       : {fixed[0][:70]}...")


if __name__ == "__main__":
    demo()

    # Non-TODO asserts: core invariants that must always hold.
    texts = [doc.text for doc in DOCUMENTS]
    emb = TfidfEmbedder().fit(texts)
    q = QA_PAIRS[1][
        "question"
    ]  # "How many chunks does a typical RAG pipeline retrieve per query?"
    sims = sorted(
        (
            (cosine_sparse(emb.embed(q), emb.embed(t)), d.id)
            for d, t in zip(DOCUMENTS, texts)
        ),
        reverse=True,
    )
    assert sims[0][1] == "rag-overview", sims[0]
    assert 0 <= cosine_dense(HashEmbedder().embed("a"), HashEmbedder().embed("b")) <= 1
    assert all(c.parent_id is not None for c in chunk_corpus())

    print(
        "\n[practice] TODOs remain: (see docstring) implement your own idf + cosine in a scratch copy"
    )
