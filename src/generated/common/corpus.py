"""Shared corpus and golden QA pairs for all practice scripts.

The corpus is a set of short, topic-distinct documentation paragraphs about
AI engineering concepts. The vocabulary is deliberately chosen so that:

- lexical retrieval (BM25) succeeds on keyword-matching questions,
- paraphrased questions require thinking about embeddings/similarity,
- metadata (section, tags) supports filtering and self-query demos,
- QA pairs contain facts (numbers, lists) that make evals checkable.

This module is stdlib-only and deterministic: importing it has no side effects.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """A single corpus document with retrieval metadata."""

    id: str
    title: str
    section: str
    tags: tuple[str, ...]
    text: str


DOCUMENTS: list[Document] = [
    Document(
        id="rag-overview",
        title="Retrieval-Augmented Generation",
        section="fundamentals",
        tags=("retrieval", "generation"),
        text=(
            "Retrieval-Augmented Generation (RAG) is an architecture that grounds "
            "a language model's answers in documents fetched at query time. Instead "
            "of relying only on parametric memory, the system retrieves relevant "
            "passages, places them into the prompt, and asks the model to answer "
            "from that context. RAG reduces hallucination because every claim can "
            "be traced back to a retrieved source, and it keeps knowledge fresh "
            "without retraining the model. A typical RAG pipeline retrieves "
            "between 5 and 20 chunks per query."
        ),
    ),
    Document(
        id="chunking",
        title="Chunking Strategies",
        section="ingestion",
        tags=("retrieval", "preprocessing"),
        text=(
            "Chunking splits long documents into pieces small enough to embed and "
            "retrieve. A common chunk size is 256 to 512 tokens with 10 to 20 "
            "percent overlap between neighbours, which stops sentences from being "
            "cut in half at a boundary. Recursive splitting first tries to break on "
            "paragraph breaks, then sentences, then words, so chunks follow the "
            "document's natural structure. Overlapping chunks trade a larger index "
            "for fewer missed passages that straddle a boundary."
        ),
    ),
    Document(
        id="embeddings",
        title="Embedding Models",
        section="fundamentals",
        tags=("retrieval", "models"),
        text=(
            "An embedding model maps text into a fixed-length vector, for example "
            "384 or 768 dimensions, such that semantically similar texts land close "
            "together under cosine similarity. Most embedding models are "
            "bi-encoders: they encode the query and each document independently, "
            "which makes vectors cacheable but blind to how the query and document "
            "interact. Cosine similarity on normalized vectors reduces to a dot "
            "product and ranges from -1 to 1."
        ),
    ),
    Document(
        id="vector-db",
        title="Vector Databases",
        section="infrastructure",
        tags=("infrastructure", "retrieval"),
        text=(
            "A vector database stores embeddings with their metadata and answers "
            "nearest-neighbour queries. Exact search compares the query against "
            "every vector, which is linear in collection size, so production "
            "indexes use approximate nearest neighbour (ANN) algorithms such as "
            "HNSW graphs or IVF clusters that trade a little recall for orders of "
            "magnitude lower latency. Most vector stores also support metadata "
            "filtering, which combines similarity search with structured "
            "predicates like section or date."
        ),
    ),
    Document(
        id="hybrid-search",
        title="Hybrid Search",
        section="retrieval",
        tags=("retrieval",),
        text=(
            "Hybrid search combines sparse lexical retrieval, usually BM25, with "
            "dense vector retrieval. BM25 matches exact terms and handles rare "
            "keywords and identifiers well, while dense retrieval captures "
            "paraphrase and synonymy. Reciprocal Rank Fusion (RRF) merges the two "
            "ranked lists by summing 1 divided by the rank plus a constant of 60 "
            "for each document, which needs no score calibration between systems. "
            "Hybrid search consistently outperforms either method alone on mixed "
            "corpora."
        ),
    ),
    Document(
        id="reranking",
        title="Cross-Encoder Re-ranking",
        section="retrieval",
        tags=("quality", "retrieval"),
        text=(
            "A cross-encoder reranker scores the query and one candidate document "
            "together in a single forward pass, so it sees token-level interaction "
            "that a bi-encoder cannot. Because that joint pass is expensive, "
            "rerankers typically process only the top 50 candidates from "
            "first-stage retrieval and shrink them to a final 5. Re-ranking adds a "
            "second model and latency to the pipeline, so it is applied after "
            "retrieval rather than instead of it."
        ),
    ),
    Document(
        id="agents",
        title="Agent Loops and Graphs",
        section="orchestration",
        tags=("agents",),
        text=(
            "An agent gives a language model control of a loop: the model chooses "
            "a tool, an executor runs it, the observation is appended to the "
            "message history, and the model runs again. The loop needs explicit "
            "termination conditions, such as a final-answer action, a maximum "
            "iteration count, or a stop token, because a model that can always "
            "call one more tool will. Agent graphs generalize the loop into nodes "
            "and conditional edges so workflows can branch and retry."
        ),
    ),
    Document(
        id="evals",
        title="RAG Evaluation",
        section="quality",
        tags=("evaluation",),
        text=(
            "A RAG evaluation needs a golden set of question, answer, and source "
            "triples, plus separate metrics for each stage. Retrieval is scored "
            "with recall@k, the fraction of relevant chunks found in the top k, "
            "while generation is scored with exact match or token-level F1 against "
            "the reference answer. Faithfulness checks that every claim in the "
            "answer is supported by the retrieved context, often with an LLM judge. "
            "Evaluations must be run on a fixed dataset so scores are comparable "
            "over time."
        ),
    ),
    Document(
        id="tokenization",
        title="Byte-Pair Encoding",
        section="fundamentals",
        tags=("preprocessing", "models"),
        text=(
            "Byte-Pair Encoding (BPE) builds a subword vocabulary by repeatedly "
            "merging the most frequent adjacent symbol pair in the corpus, "
            "starting from raw bytes. Subwords balance vocabulary size against "
            "sequence length: common words stay whole, rare words split into "
            "reusable pieces, and unseen words can still be encoded, so there is "
            "no out-of-vocabulary token. A typical BPE vocabulary for a "
            "production model holds between 32 thousand and 128 thousand merges."
        ),
    ),
    Document(
        id="metadata",
        title="Ingestion Metadata and Self-Querying",
        section="ingestion",
        tags=("infrastructure",),
        text=(
            "Ingestion attaches metadata to every chunk before it is indexed, "
            "such as the source document, section, author, and timestamp. "
            "Self-querying extracts a structured filter from a natural-language "
            "question, for example the phrase from the performance section turns "
            "into an equality filter on the section field, and pushes that filter "
            "into the vector store so similarity search only runs over matching "
            "rows."
        ),
    ),
]

# Golden set used by Day 5 evals (and Day 3 retrieval tests).
# Each pair maps a question to a reference answer that is (nearly) verbatim in
# the relevant document, so an extractive mock LLM can score well -- but note
# that a few questions are phrased as paraphrases to expose retrieval misses.
QA_PAIRS: list[dict[str, str]] = [
    {
        "question": "What does RAG stand for and what problem does it solve?",
        "answer": "RAG reduces hallucination because every claim can be traced back to a retrieved source",
        "doc_id": "rag-overview",
    },
    {
        "question": "How many chunks does a typical RAG pipeline retrieve per query?",
        "answer": "A typical RAG pipeline retrieves between 5 and 20 chunks per query",
        "doc_id": "rag-overview",
    },
    {
        "question": "What chunk size is common, and with how much overlap?",
        "answer": "A common chunk size is 256 to 512 tokens with 10 to 20 percent overlap between neighbours",
        "doc_id": "chunking",
    },
    {
        "question": "How many dimensions do embedding models use?",
        "answer": "for example 384 or 768 dimensions",
        "doc_id": "embeddings",
    },
    {
        "question": "Which approximate nearest neighbour algorithms do production vector indexes use?",
        "answer": "HNSW graphs or IVF clusters",
        "doc_id": "vector-db",
    },
    {
        "question": "What constant does Reciprocal Rank Fusion use?",
        "answer": "a constant of 60",
        "doc_id": "hybrid-search",
    },
    {
        "question": "How many candidates do rerankers typically process?",
        "answer": "only the top 50 candidates",
        "doc_id": "reranking",
    },
    {
        "question": "Why does an agent loop need termination conditions?",
        "answer": "because a model that can always call one more tool will",
        "doc_id": "agents",
    },
    {
        "question": "What does recall@k measure?",
        "answer": "the fraction of relevant chunks found in the top k",
        "doc_id": "evals",
    },
    {
        "question": "What vocabulary size does a typical production BPE model have?",
        "answer": "between 32 thousand and 128 thousand merges",
        "doc_id": "tokenization",
    },
    {
        "question": "What is self-querying?",
        "answer": "Self-querying extracts a structured filter from a natural-language question",
        "doc_id": "metadata",
    },
    {
        "question": "What are the two retrieval methods combined in hybrid search?",
        "answer": "sparse lexical retrieval, usually BM25, with dense vector retrieval",
        "doc_id": "hybrid-search",
    },
]


def full_text() -> str:
    """All documents concatenated with paragraph breaks, for tokenization."""
    return "\n\n".join(doc.text for doc in DOCUMENTS)


if __name__ == "__main__":
    print(f"{len(DOCUMENTS)} documents, {len(QA_PAIRS)} golden QA pairs")
    for doc in DOCUMENTS:
        print(f"  {doc.id:<15} section={doc.section:<14} words={len(doc.text.split())}")
