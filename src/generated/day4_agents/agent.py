"""Day 4b: Agent tool loop + state graph with conditional edges.

Interview talking points:
- The agent loop: model -> tool call -> observation -> model, repeat. The
  executor owns the loop; the model only ever picks the *next* action.
- Termination: a final-answer action, a max-iteration guard, or a
  no-progress rule. Without one, a loop that can always call a tool, will.
- The loop generalizes to a graph: nodes do work, conditional edges route,
  and cycles allow retry workflows (plan -> search -> grade -> retry/answer).

Run:  uv run src/generated/day4_agents/agent.py
"""

from dataclasses import dataclass, field
from typing import Callable

from src.generated.common.corpus import DOCUMENTS, QA_PAIRS
from src.generated.common.mock_llm import ScriptedAgentLLM, content_terms
from src.generated.day3_rag.hybrid_search import build_index, hybrid_search

MAX_ITERATIONS = 5


# ------------------------------------------------------------------- tools ---


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[dict], str]


def make_tools(index) -> list[Tool]:
    """`search`: hybrid RAG retrieval. `fetch_document`: read a full doc by id."""
    bm25, embedder, chunks = index

    def search(args: dict) -> str:
        hits = hybrid_search(args["query"], bm25, embedder, chunks, k=3)
        if not hits:
            return "NO_RESULTS"
        lines = [f"doc_id:{hits[0].doc_id}", *[f"  - {h.text}" for h in hits]]
        return "\n".join(lines)

    def fetch_document(args: dict) -> str:
        doc = next((d for d in DOCUMENTS if d.id == args["doc_id"]), None)
        return doc.text if doc else f"doc {args['doc_id']} not found"

    return [
        Tool(
            name="search",
            description="hybrid RAG retrieval over the corpus",
            func=search,
        ),
        Tool(
            name="fetch_document",
            description="fetch a full document by id",
            func=fetch_document,
        ),
    ]


# -------------------------------------------------------------- agent loop ---


@dataclass
class Agent:
    llm: ScriptedAgentLLM
    tools: dict[str, Tool]
    max_iterations: int = MAX_ITERATIONS
    trace: list[str] = field(default_factory=list)

    def run(self, question: str) -> str:
        observations: list[str] = []
        for step in range(1, self.max_iterations + 1):
            action = self.llm.decide(question, observations)
            if action["type"] == "final":
                self.trace.append(f"step {step}: FINAL -> {action['answer'][:50]}...")
                return action["answer"]
            tool = self.tools[action["tool"]]
            result = tool.func(action["args"])
            self.trace.append(
                f"step {step}: {tool.name}({action['args']}) -> {result[:40]}..."
            )
            if result == "NO_RESULTS":
                return "I don't know."  # terminate honestly instead of looping
            observations.append(result)
        self.trace.append(
            f"step {self.max_iterations}: MAX_ITERATIONS reached, stopping"
        )
        return "I couldn't finish within the iteration budget."


# ------------------------------------------------------- mini state graph ---


@dataclass
class Graph:
    """LangGraph-in-miniature: named nodes + conditional edges, with cycles."""

    nodes: dict[str, Callable] = field(default_factory=dict)
    edges: dict[str, str | None] = field(default_factory=dict)  # fixed edges
    conditional: dict[str, Callable] = field(default_factory=dict)  # routers
    max_steps: int = 6

    def add_node(self, name: str, fn: Callable) -> None:
        self.nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self.edges[src] = dst

    def add_conditional_edges(self, src: str, router: Callable, mapping: dict) -> None:
        self.conditional[src] = lambda state: mapping[router(state)]

    def invoke(self, state: dict, entry: str) -> dict:
        current = entry
        for step in range(1, self.max_steps + 1):
            state = dict(state)
            state["trace"] = state.get("trace", []) + [f"step {step}: node '{current}'"]
            state["path"] = state.get("path", []) + [current]
            state = self.nodes[current](state)
            if state.get("__end__"):
                state["trace"].append("terminal")
                state["path"].append("END")
                return state
            if current in self.conditional:
                current = self.conditional[current](state)
            else:
                current = self.edges[current]
        state["trace"] = state.get("trace", []) + ["MAX_STEPS reached"]
        state["path"] = state.get("path", []) + ["MAX_STEPS"]
        state["final_answer"] = "I couldn't finish within the graph step budget."
        return state


def build_rag_graph(index) -> Graph:
    """plan -> search -> grade -> (answer | search-with-relaxed-query ...)

    The retry cycle is the point: a workflow that can *decide* to try
    again with a different query, with a hard step budget to stop it
    from running forever.
    """
    bm25, embedder, chunks = index
    attempts: list[str] = []

    def plan(state):
        attempts.clear()
        state["query"] = " ".join(content_terms(state["question"])[:6])
        return state

    def search(state):
        attempts.append(state["query"])
        hits = hybrid_search(state["query"], bm25, embedder, chunks, k=3)
        state["contexts"] = [h.text for h in hits]
        return state

    def grade(state):
        # rule-based relevance check (an LLM grader in production)
        q_terms = set(content_terms(state["question"]))
        state["best_overlap"] = max(
            (len(q_terms & set(content_terms(c))) for c in state["contexts"]), default=0
        )
        return state

    def router(state) -> str:
        if state["best_overlap"] >= 3 or len(attempts) >= 2:
            return "done"
        # relax the query: shorter, more generic terms, then cycle back to search
        state["query"] = (
            " ".join(content_terms(state["question"])[:4]) + " overview definition"
        )
        return "retry"

    def answer(state):
        from src.generated.common.mock_llm import extractive_answer

        state["final_answer"] = extractive_answer(state["question"], state["contexts"])
        state["__end__"] = True
        return state

    g = Graph()
    g.add_node("plan", plan)
    g.add_node("search", search)
    g.add_node("grade", grade)
    g.add_node("answer", answer)
    g.add_edge("plan", "search")
    g.add_edge("search", "grade")
    g.add_conditional_edges("grade", router, {"done": "answer", "retry": "search"})
    g.add_edge("answer", "__end__")
    return g


# -------------------------------------------------------------------- demo ---


def demo() -> None:
    index = build_index()
    tools = {t.name: t for t in make_tools(index)}

    print("== Agent tool loop (ReAct-style) ==")
    qa = QA_PAIRS[1]
    agent = Agent(llm=ScriptedAgentLLM(), tools=tools)
    answer = agent.run(qa["question"])
    print(f"question: {qa['question']}")
    for line in agent.trace:
        print(f"  {line}")
    print(f"answer  : {answer}")
    assert len(agent.trace) <= MAX_ITERATIONS, "max-iteration guard failed"

    print("\n== State graph with a retry cycle ==")
    g = build_rag_graph(index)
    for q in (
        QA_PAIRS[2]["question"],
        "How do you keep a model from making things up?",
    ):
        state = g.invoke({"question": q}, entry="plan")
        print(f"question: {q}")
        print(f"  path  : {' -> '.join(state['path'])}")
        print(f"  answer: {state['final_answer'][:70]}")
        print()

    print("== What terminates a loop ==")
    print(
        f"  final-answer action, MAX_ITERATIONS={MAX_ITERATIONS}, NO_RESULTS rule, MAX_STEPS={g.max_steps}"
    )


if __name__ == "__main__":
    demo()

    # Non-TODO asserts: the loop must always terminate within budget.
    index = build_index()
    tools = {t.name: t for t in make_tools(index)}
    for qa in QA_PAIRS:
        agent = Agent(llm=ScriptedAgentLLM(), tools=tools)
        agent.run(qa["question"])
        assert len(agent.trace) <= MAX_ITERATIONS
    g = build_rag_graph(index)
    for qa in QA_PAIRS:
        state = g.invoke({"question": qa["question"]}, entry="plan")
        assert len(state["trace"]) <= g.max_steps + 1
    print("[agent asserts passed]")
