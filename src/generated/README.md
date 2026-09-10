# AI Engineer Interview Prep — Practice Scripts

From-scratch reference implementations of the five core interview topics.
Everything runs offline, stdlib-only, deterministically (a mock LLM stands in
for a hosted model, so every score and loop trace is reproducible).

These live in `src/generated/` as **reference implementations** — study and run
them, then hand-write your own versions (e.g. in `src/practice/`) from memory.

## Topics

### Day 1 — `day1_tokenization/bpe.py`
- How does BPE training work? (merge the most frequent adjacent pair, repeat)
- Why subwords? (vocab size vs sequence length, no OOV)
- Why do we pre-tokenize before BPE? (merges must not cross word boundaries)
- Encode: replay merges by rank, not time — why it matters.
- The `</w>` end-of-word marker (GPT-2's `Ġ`).

### Day 2 — `day2_embeddings/embeddings.py`
- TF-IDF: why idf rewards *distinctive* terms; smoothed idf formula.
- Cosine vs euclidean; length normalization.
- Bi-encoder (embed separately) vs cross-encoder (score jointly) — when each.
- What a real embedding model adds over bag-of-words (our hash embedder is
  deliberately semantic-blind: the gap is the interview answer).
- Chunking: fixed + overlap, recursive structure-aware, parent-child.

### Day 3 — `day3_rag/vector_store.py` + `hybrid_search.py`
- Vector DB interface: upsert, top-k by cosine, metadata filtering (pre vs post).
- Exact vs ANN (HNSW/IVF) search: the recall/latency trade-off.
- BM25: tf saturation (k1), length normalization (b), idf.
- Hybrid search: why keyword and dense retrieval fail differently; RRF fusion
  (rank-only, k=60, no score calibration).

### Day 4 — `day3_rag/pipeline.py` + `day4_agents/agent.py`
- Advanced RAG patterns: query rewriting, re-ranking, contextual compression,
  self-querying (structured filters from natural language), parent-document
  retrieval.
- The agent loop: model → tool → observation → repeat; who owns the loop.
- Termination: final answer, max iterations, no-progress rules.
- State graphs: nodes, conditional edges, cycles for retry workflows
  (LangGraph-in-miniature).

### Day 5 — `day5_evals/eval.py`
- Why retrieval and generation must be evaluated separately.
- recall@k, MRR (retrieval); exact match, token F1 (generation).
- Faithfulness / groundedness and the LLM-as-judge pattern (mocked here).
- Reading a scorecard: which metric points at which pipeline stage.

## Layout

```
src/generated/
├── common/            corpus.py (docs + golden QA), mock_llm.py
├── day1_tokenization/   bpe.py
├── day2_embeddings/     embeddings.py (TF-IDF, hash embedder, chunking)
├── day3_rag/            vector_store.py, hybrid_search.py, pipeline.py
├── day4_agents/         agent.py (tool loop + state graph)
└── day5_evals/          eval.py (metrics + scorecard)
```

Each script imports the previous days' work: chunks come from Day 2, the
index from Day 3, the pipeline from Day 4, and the scorecard grades it all.
