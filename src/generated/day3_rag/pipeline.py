"""Day 4a: RAG pipeline -- naive vs advanced patterns.

Architecture patterns demonstrated (each is a named function):
- query rewriting      : expand acronyms/synonyms for the dense leg
- hybrid retrieval     : BM25 + dense + RRF (from Day 3)
- re-ranking           : second-stage scorer over the top candidates
- contextual compression: keep only query-relevant sentences per chunk
- self-querying        : parse a structured metadata filter out of the question
- parent-document      : index small chunks, return their larger parent

Run:  uv run src/generated/day3_rag/pipeline.py
"""

from src.generated.common.corpus import DOCUMENTS, QA_PAIRS
from src.generated.common.mock_llm import extractive_answer
from src.generated.day2_embeddings.embeddings import (
    TfidfEmbedder,
    cosine_dense,
    cosine_sparse,
    tokenize,
)
from src.generated.day3_rag.hybrid_search import build_index, hybrid_search

# ------------------------------------------------------------------ rewrite --

ACRONYMS = {
    "rag": "Retrieval-Augmented Generation",
    "ann": "approximate nearest neighbour",
    "bpe": "Byte-Pair Encoding",
    "rrf": "Reciprocal Rank Fusion",
    "llm": "language model",
    "ivf": "inverted file clusters",
    "hnsw": "hierarchical navigable small world graphs",
}


def rewrite_query(query: str) -> str:
    """Expand acronyms so the dense leg 'sees' their surface forms too.

    Real systems use an LLM for this (add synonyms, resolve coreference,
    decompose multi-hop questions). A rule-based map keeps us offline.
    """
    extras = []
    lowered = query.lower()
    for acro, expansion in ACRONYMS.items():
        if acro in lowered and expansion.lower() not in lowered:
            extras.append(expansion)
    return f"{query} {' '.join(extras)}".strip()


# ------------------------------------------------------------------ rerank ---


def make_reranker(chunks: list):
    """Second-stage scorer: TF-IDF cosine (a stand-in for a cross-encoder).

    A real cross-encoder scores (query, document) jointly in one forward
    pass and sees token interactions; TF-IDF cosine is the same interface
    with a cheap scorer inside.
    """
    embedder = TfidfEmbedder().fit([c.text for c in chunks])

    def rerank(query: str, retrieved, top_k: int = 5):
        qv = embedder.embed(query)
        scored = sorted(
            (
                (cosine_sparse(qv, embedder.embed(r.text)), -i, r)
                for i, r in enumerate(retrieved)
            ),
            key=lambda t: (-t[0], t[1]),
        )
        return [r for _, _, r in scored[:top_k]]

    return rerank


# ---------------------------------------------------- contextual compression --


def compress_context(
    query: str, retrieved, max_sentences_per_chunk: int = 1
) -> list[str]:
    """Keep only the sentences of each chunk that overlap the query.

    The prompt is where token budget is spent; dropping irrelevant
    sentences both saves budget and removes distractors that would pull
    the answer away from the right passage.
    """
    import re

    q_terms = set(tokenize(query))
    out = []
    for r in retrieved:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", r.text) if s.strip()]
        scored = [
            (len(q_terms & set(tokenize(s))), -i, s) for i, s in enumerate(sentences)
        ]
        scored.sort(reverse=True)
        picked = [s for score, _, s in scored[:max_sentences_per_chunk] if score > 0]
        out.extend(picked or ([sentences[0]] if sentences else []))
    return out


# ------------------------------------------------------------- self-query ---


def self_query(query: str) -> dict | None:
    """Extract a metadata filter from natural language, e.g.
    'in the retrieval section' -> {'section': 'retrieval'}."""
    import re

    m = re.search(r"(?:in|from|under)(?: the)? ([a-z]+) section", query.lower())
    sections = {doc.section for doc in DOCUMENTS}
    if m and m.group(1) in sections:
        return {"section": m.group(1)}
    return None


# --------------------------------------------------- parent-document search --


def parent_document_search(query: str, index, top_k: int = 3):
    """Search child chunks, then return the (larger) parent text."""
    bm25, embedder, chunks = index
    children = hybrid_search(query, bm25, embedder, chunks, k=top_k * 2)

    # parent_id -> text: rebuild the parent from its id "doc:N"
    parents: dict[str, str] = {}
    for c in children:
        parent_id = c.metadata.get("parent_id") or f"{c.doc_id}:0"
        if parent_id not in parents:
            doc = next(d for d in DOCUMENTS if d.id == c.doc_id)
            idx = int(parent_id.split(":")[1])
            words = doc.text.split()
            parents[parent_id] = " ".join(words[idx * 48 : (idx + 1) * 48])

    # one parent per hit, ordered by the best child
    seen: list[str] = []
    for c in children:
        pid = c.metadata.get("parent_id") or f"{c.doc_id}:0"
        if pid not in [s for s in seen]:
            seen.append(pid)
    return [parents[pid] for pid in seen[:top_k]], children


# ------------------------------------------------------- naive vs advanced ---


def naive_rag(question: str, index) -> dict:
    """Dense-only top-3, full chunks into the prompt, answer."""
    bm25, embedder, chunks = index
    qv = embedder.embed(question)
    ranked = sorted(
        (
            (cosine_dense(qv, embedder.embed(c.text)), -i, c)
            for i, c in enumerate(chunks)
        ),
        key=lambda t: (-t[0], t[1]),
    )
    top = ranked[:3]
    contexts = [c.text for _, _, c in top]
    return {
        "answer": extractive_answer(question, contexts),
        "contexts": contexts,
        "retrieved_doc_ids": [c.doc_id for _, _, c in top],
        "trace": ["dense top-3 -> stuff full chunks -> answer"],
    }


def advanced_rag(question: str, index, rerank) -> dict:
    """rewrite -> hybrid -> rerank -> compress -> answer, with an optional
    self-query filter and parent-document expansion."""
    steps: list[str] = []
    rewritten = rewrite_query(question)
    if rewritten != question:
        steps.append(f"rewrite: {rewritten}")

    bm25, embedder, chunks = index
    filter = self_query(question)
    if filter:
        steps.append(f"self-query filter: {filter}")

    # filter chunks before search when we have a structured constraint
    pool = chunks
    if filter:
        pool = [c for c in chunks if c.metadata.get("section") == filter["section"]]

    candidates = hybrid_search(rewritten, bm25, embedder, pool, k=8)
    steps.append(f"hybrid retrieve: {len(candidates)} candidates")

    top = rerank(question, candidates, top_k=5)
    steps.append(f"rerank: kept {len(top)}")

    contexts = compress_context(question, top)
    steps.append(f"compress: {sum(len(c.split()) for c in contexts)} words in prompt")

    # order-preserving unique doc ids (a plain set's iteration order is
    # process-dependent, which would make MRR non-deterministic)
    unique_doc_ids: list[str] = []
    for c in top:
        if c.doc_id not in unique_doc_ids:
            unique_doc_ids.append(c.doc_id)

    return {
        "answer": extractive_answer(question, contexts),
        "contexts": contexts,
        "retrieved_doc_ids": unique_doc_ids,
        "trace": steps,
    }


# -------------------------------------------------------------------- demo ---


def demo() -> None:
    index = build_index()
    rerank = make_reranker(index[2])

    print("== Naive vs advanced RAG ==")
    for qa in QA_PAIRS[:4]:
        print(f"\nQ: {qa['question']}")
        n = naive_rag(qa["question"], index)
        a = advanced_rag(qa["question"], index, rerank)
        for line in a["trace"]:
            print(f"   | {line}")
        hit = lambda r: "HIT " if qa["doc_id"] in r["retrieved_doc_ids"] else "MISS"
        print(f"   naive   [{hit(n)}] {n['answer'][:70]}")
        print(f"   advanced[{hit(a)}] {a['answer'][:70]}")

    print("\n== Parent-document retrieval (small-to-big) ==")
    parents, children = parent_document_search("what overlap do chunks need?", index)
    for p in parents:
        print(f"  parent ({len(p.split())}w): {p[:70]}...")
    print(f"  (children hit precisely, parents carry the surrounding context)")

    print("\n== Self-querying ==")
    q = "What is evaluated, in the quality section?"
    print(f"  query : {q}")
    print(f"  filter: {self_query(q)}")
    a = advanced_rag(q, index, rerank)
    print(f"  docs  : {a['retrieved_doc_ids']}")


if __name__ == "__main__":
    demo()

    # Non-TODO asserts.
    index = build_index()
    assert (
        rewrite_query("What is RAG?") == "What is RAG? Retrieval-Augmented Generation"
    )
    assert self_query("anything in the retrieval section?") == {"section": "retrieval"}
    assert self_query("no sections here") is None

    rerank = make_reranker(index[2])
    qa = QA_PAIRS[1]
    a = advanced_rag(qa["question"], index, rerank)
    assert "rag-overview" in a["retrieved_doc_ids"], a["retrieved_doc_ids"]
    assert "5 and 20" in a["answer"], a["answer"]

    print("\n[pipeline asserts passed]")
