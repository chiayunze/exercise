"""Deterministic mock LLM.

Stands in for a hosted model so every demo runs offline and every eval score
is reproducible. Two behaviours:

- ``extractive_answer``: given a question and retrieved contexts, pick the
  single sentence that shares the most content words with the question. This
  makes retrieval quality directly observable: bad retrieval -> bad answer,
  because the "model" is only as good as the context it is given.
- ``ScriptedAgentLLM``: a rule-based policy for the Day 4 agent loop. It
  issues a search, then a document fetch, then a final answer -- demonstrating
  multi-tool loops with explicit termination.
"""

import re

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
    "did",
    "be",
    "been",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "as",
    "by",
    "from",
    "at",
    "into",
    "than",
    "then",
    "so",
    "such",
    "can",
    "will",
    "would",
    "should",
    "have",
    "has",
    "had",
    "not",
    "but",
    "you",
    "we",
    "they",
}

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def content_terms(text: str) -> list[str]:
    """Lowercase tokens with stopwords removed -- the unit of all overlap math."""
    return [t for t in _WORD_RE.findall(text.lower()) if t not in STOPWORDS]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def extractive_answer(question: str, contexts: list[str]) -> str:
    """Grounded answer: the context sentence best matching the question.

    A real LLM would synthesize; this one quotes. The point for interview
    prep: the *retrieval* is what carries the answer quality here, and Day 5
    evals can measure that without an API key.
    """
    if not contexts:
        return "I don't have enough context to answer that question."

    q_terms = set(content_terms(question))
    best_sentence, best_score = None, -1
    for context in contexts:
        for sentence in _sentences(context):
            score = len(q_terms & set(content_terms(sentence)))
            if score > best_score:
                best_sentence, best_score = sentence, score

    if best_sentence is None or best_score == 0:
        # No lexical overlap anywhere: fall back to the first sentence.
        return _sentences(contexts[0])[0]
    return best_sentence


class ScriptedAgentLLM:
    """Rule-based stand-in for an agent's model.

    Policy (deterministic on the observation count):
      1st turn -> call ``search`` with the question's content terms
      2nd turn -> call ``fetch_document`` for the top search hit
      3rd turn -> emit a final answer built from everything observed

    Any turn with zero search hits terminates immediately with an honest
    "I don't know" instead of looping -- the same discipline a real agent
    needs.
    """

    def decide(self, question: str, observations: list[str]) -> dict:
        """Return an action dict: tool call or final answer."""
        if not observations:
            query = " ".join(content_terms(question)[:6])
            return {"type": "tool", "tool": "search", "args": {"query": query}}

        if len(observations) == 1:
            top_id = observations[0].splitlines()[0].split(":")[1].strip()
            return {
                "type": "tool",
                "tool": "fetch_document",
                "args": {"doc_id": top_id},
            }

        text = "\n".join(observations)
        answer = extractive_answer(question, [text])
        return {"type": "final", "answer": answer}


if __name__ == "__main__":
    # Tiny smoke test.
    q = "How many chunks does a typical RAG pipeline retrieve per query?"
    ctx = [
        "A typical RAG pipeline retrieves between 5 and 20 chunks per query. "
        "Retrieval-Augmented Generation grounds answers in documents fetched at query time."
    ]
    print("question  :", q)
    print("answer    :", extractive_answer(q, ctx))

    llm = ScriptedAgentLLM()
    print("turn 1    :", llm.decide(q, []))
    obs = [
        "doc_id:rag-overview\nA typical RAG pipeline retrieves between 5 and 20 chunks per query."
    ]
    print("turn 2    :", llm.decide(q, obs))
