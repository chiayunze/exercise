"""Day 1: Byte-Pair Encoding, from scratch.

Interview talking points:
- Why subwords? Word-level vocab explodes and cannot handle unseen words;
  char-level sequences are too long. BPE lands in between: frequent words
  stay whole, rare words split into reusable pieces -> no OOV token.
- Training: repeatedly merge the most frequent adjacent symbol pair.
- Encoding: apply merges in learned order (lowest rank first).
- Cost angle: tokens, not words, are what LLM APIs bill and what context
  windows count.

Run:  uv run src/generated/day1_tokenization/bpe.py
"""

from collections import Counter

from src.generated.common.corpus import full_text

# Words split into symbols; </w> marks the end of a word so merges never
# cross word boundaries (same trick GPT-2 uses with its byte-level marker).
END = "</w>"

# Approximation of GPT-2's pre-tokenization regex: words, or a single
# punctuation character. Pre-tokenization runs *before* BPE.
WORD_RE = r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]"


def pre_tokenize(text: str) -> list[str]:
    import re

    return re.findall(WORD_RE, text)


def word_frequencies(text: str) -> Counter[tuple[str, ...]]:
    """Map each word to a symbol tuple (chars + END marker), with counts."""
    freqs: Counter[tuple[str, ...]] = Counter()
    for word in pre_tokenize(text):
        freqs[tuple(word) + (END,)] += 1
    return freqs


def count_pairs(freqs: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
    """Count adjacent symbol pairs, weighted by word frequency."""
    pairs: Counter[tuple[str, str]] = Counter()
    for symbols, count in freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += count
    return pairs


def apply_merge(pair: tuple[str, str], symbols: tuple[str, ...]) -> tuple[str, ...]:
    """Replace every occurrence of `pair` inside `symbols` with its join."""
    merged = pair[0] + pair[1]
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(text: str, num_merges: int) -> list[tuple[str, str]]:
    """Learn `num_merges` merges; returns them in learned order (= rank)."""
    freqs = word_frequencies(text)
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs = count_pairs(freqs)
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))  # tie-break deterministically
        freqs = Counter({apply_merge(best, w): c for w, c in freqs.items()})
        merges.append(best)
    return merges


def encode_word(word: str, merge_rank: dict[tuple[str, str], int]) -> list[str]:
    """Encode one word: repeatedly merge its lowest-rank adjacent pair.

    This is the classic GPT-2 style encoder: it does NOT replay merges
    chronologically; it always picks the applicable pair that was learned
    earliest, which is what makes encoding match training segmentation.
    """
    symbols: list[str] = list(word) + [END]
    while len(symbols) > 1:
        pairs = [
            (merge_rank[(a, b)], (a, b))
            for a, b in zip(symbols, symbols[1:])
            if (a, b) in merge_rank
        ]
        if not pairs:
            break
        _, (a, b) = min(pairs)
        merged = a + b
        out: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


def encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    """Encode a full text into subword tokens."""
    merge_rank = {pair: rank for rank, pair in enumerate(merges)}
    tokens: list[str] = []
    for word in pre_tokenize(text):
        tokens.extend(encode_word(word, merge_rank))
    return tokens


def decode(tokens: list[str]) -> str:
    """Reassemble text: tokens keep the END marker, so word boundaries survive."""
    return "".join(tokens).replace(END, " ").strip()


def vocab_size(merges: list[tuple[str, str]], text: str) -> int:
    """Base characters + learned merges (simplified; real BPE adds 256 bytes)."""
    return len({ch for ch in text if not ch.isspace()}) + len(merges)


def demo() -> None:
    text = full_text()
    merges = train_bpe(text, num_merges=60)

    print("== BPE training ==")
    print(f"corpus: {len(pre_tokenize(text))} words, {len(text)} characters")
    print(f"base char vocab : {len({c for c in text if not c.isspace()})}")
    print(f"after 60 merges : {vocab_size(merges, text)}")
    print(f"first 10 merges : {[a + b for a, b in merges[:10]]}")

    sample = "Retrieval retrieves retrieved chunks, and unblendability too."
    tokens = encode(sample, merges)
    print("\n== Encoding an unseen sentence ==")
    print(f"  text   : {sample}")
    print(f"  tokens : {tokens}")
    print(f"  (the {END!r} suffix marks word ends, like GPT-2's 'G' marker)")
    print(
        f"  chars  : {len(sample)}  words: {len(sample.split())}  tokens: {len(tokens)}"
    )

    ratio = len(sample) / len(tokens)
    print(f"  compression: {ratio:.2f} chars/token vs 1.0 for char-level")

    unseen = "tokenization"
    print(
        f"\n== Unseen word '{unseen}' splits into: {encode_word(unseen, {p: i for i, p in enumerate(merges)})}"
    )
    print("   (no OOV: any string encodes into char/subword pieces)")


# ---------------------------------------------------------------- practice ---
# Solve these yourself, then run the asserts at the bottom of the file.


def encode_sequential(text: str, merges: list[tuple[str, str]]) -> list[str]:
    """TODO: alternative encoder -- replay merges in chronological order
    (for each merge in turn, apply it to every word) instead of picking
    the lowest-rank pair. Return the flat token list."""
    raise NotImplementedError("practice: implement me")


def token_count_api_bill(text: str, merges: list[tuple[str, str]]) -> int:
    """TODO: return how many tokens a model with this BPE vocab would bill
    for `text` (hint: just len(encode(...)))."""
    raise NotImplementedError("practice: implement me")


if __name__ == "__main__":
    demo()

    _text = full_text()
    _merges = train_bpe(_text, num_merges=60)

    # Reference encoder must be idempotent over its own vocabulary.
    assert decode(encode(_text, _merges)).split() == pre_tokenize(_text)

    # A merge learned from the corpus must apply somewhere in the corpus.
    assert any("th" in tok for tok in encode(_text, _merges))

    print("\n[practice] TODOs remain: encode_sequential, token_count_api_bill")
