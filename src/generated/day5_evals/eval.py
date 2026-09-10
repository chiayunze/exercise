"""Day 5: Evals -- golden set, retrieval + generation metrics, faithfulness.

Interview talking points:
- Evaluate retrieval and generation SEPARATELY: a wrong answer with perfect
  retrieval is a generation bug; a wrong answer with bad retrieval is an
  index/query bug. One end-to-end number hides which side to fix.
- recall@k (did we find the right chunk?), MRR (how high does it rank?),
  exact match + token F1 (generation), faithfulness (is every claim
  supported by the retrieved context?).
- Fixed dataset = comparable scores over time. Run the same golden set
  after every pipeline change.

Run:  uv run src/generated/day5_evals/eval.py
"""

from src.generated.common.corpus import DOCUMENTS, QA_PAIRS
from src.generated.common.mock_llm import content_terms
from src.generated.day3_rag.hybrid_search import build_index, hybrid_search
from src.generated.day3_rag.pipeline import advanced_rag, make_reranker

# ------------------------------------------------------------ text metrics ---


def normalize(text: str) -> list[str]:
    return content_terms(text)


def exact_match(pred: str, gold: str) -> float:
    """1.0 only on a normalized string match. Harsh, but unfakeable."""
    return 1.0 if " ".join(normalize(pred)) == " ".join(normalize(gold)) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token F1: overlap of token multisets."""
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0
    from collections import Counter

    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


# -------------------------------------------------------- retrieval metrics --


def recall_at_k(retrieved_doc_ids: list[str], relevant_id: str, k: int) -> float:
    """Fraction of the (single) relevant doc found in the top k.
    Golden set has one relevant doc per question, so recall@k is 1/0."""
    return 1.0 if relevant_id in retrieved_doc_ids[:k] else 0.0


def mrr(retrieved_doc_ids: list[str], relevant_id: str) -> float:
    """Mean reciprocal rank helper: 1/rank of the first relevant hit."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == relevant_id:
            return 1.0 / rank
    return 0.0


# ----------------------------------------------------------- faithfulness ---


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Rule-based stand-in for an LLM judge: the fraction of the answer's
    content words that appear anywhere in the retrieved context. A real
    judge prompts a model with (answer, context) and asks 'is every claim
    supported?'; this lexical version is free and deterministic."""
    answer_terms = set(normalize(answer))
    if not answer_terms:
        return 0.0
    context_terms: set[str] = set()
    for c in contexts:
        context_terms.update(normalize(c))
    supported = sum(1 for t in answer_terms if t in context_terms)
    return supported / len(answer_terms)


# ---------------------------------------------------------------- scorecard --


def run_eval(k: int = 5) -> dict:
    index = build_index()
    rerank = make_reranker(index[2])

    rows = []
    for qa in QA_PAIRS:
        result = advanced_rag(qa["question"], index, rerank)
        retrieved = result["retrieved_doc_ids"]
        row = {
            "question": qa["question"],
            "recall@k": recall_at_k(retrieved, qa["doc_id"], k),
            "mrr": mrr(retrieved, qa["doc_id"]),
            "exact_match": exact_match(result["answer"], qa["answer"]),
            "token_f1": token_f1(result["answer"], qa["answer"]),
            "faithfulness": faithfulness(result["answer"], result["contexts"]),
            "answer": result["answer"],
            "gold": qa["answer"],
        }
        rows.append(row)
    return {"rows": rows, "k": k}


def print_scorecard(evals: dict) -> None:
    rows, k = evals["rows"], evals["k"]
    print(f"== Scorecard: {len(rows)} golden questions, retrieval@{k} ==\n")
    header = f"{'recall@k':>8} {'mrr':>5} {'em':>4} {'tokF1':>6} {'faith':>6}  question"
    print(header)
    for r in rows:
        print(
            f"{r['recall@k']:>8.1f} {r['mrr']:>5.2f} {r['exact_match']:>4.1f} "
            f"{r['token_f1']:>6.2f} {r['faithfulness']:>6.2f}  {r['question'][:44]}"
        )

    mean = lambda key: sum(r[key] for r in rows) / len(rows)
    print(f"\n{'-' * len(header)}")
    print(f"{'mean recall@k':>16}: {mean('recall@k'):.2f}")
    print(f"{'mean mrr':>16}: {mean('mrr'):.2f}")
    print(f"{'mean exact match':>16}: {mean('exact_match'):.2f}")
    print(f"{'mean token f1':>16}: {mean('token_f1'):.2f}")
    print(f"{'mean faithfulness':>16}: {mean('faithfulness'):.2f}")

    print("\n== Reading the scorecard ==")
    print("  - recall/mrr isolate retrieval quality (chunking, embedding, hybrid)")
    print("  - em/token_f1 isolate generation quality (prompting, extraction)")
    print("  - faithfulness catches answers unsupported by the retrieved context")
    print("  - note: exact match is harsh for an extractive 'model' that quotes whole")
    print("    sentences; token_f1 and faithfulness are the informative signals here")
    worst = min(rows, key=lambda r: r["recall@k"])
    print(f"\n  worst retrieval: {worst['question']}")
    print(f"    got : {worst['answer'][:70]}")
    print(f"    gold: {worst['gold'][:70]}")


# -------------------------------------------------------------------- demo ---

if __name__ == "__main__":
    evals = run_eval(k=5)
    print_scorecard(evals)

    # Non-TODO asserts: metric definitions behave as specified.
    assert exact_match("The cat sat.", "the CAT sat") == 1.0
    assert exact_match("cat sat", "cat stood") == 0.0
    # tokens {cat,dog,fox} vs {cat,dog,bird}: overlap 2 -> p=r=2/3 -> f1=2/3
    assert abs(token_f1("cat dog fox", "cat dog bird") - 2 / 3) < 1e-9
    assert recall_at_k(["x", "y", "z"], "y", 5) == 1.0
    assert recall_at_k(["x", "y", "z"], "y", 1) == 0.0
    assert mrr(["x", "y", "z"], "z") == 1 / 3
    assert faithfulness("cats and dogs", ["dogs live in houses"]) == 0.5

    print("\n[metric asserts passed]")
