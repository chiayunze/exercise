# Fix bugs in `src/handwritten/bpe.py`

Status: **saved for future reference — not implemented.** The user asked to keep this
plan on file and explicitly chose *not* to apply it yet ("no need to work on it").
When picked up, the confirmed scope is: fix bugs 1 and 2. The word-boundary refactor
and strict `from_bytes` change are out of scope.

## Context

`src/handwritten/bpe.py` is a from-scratch BPE practice implementation (new, untracked
file). It runs and round-trips on the demo path:

```
input 4629 encoded 3611 decoded 4629
bytes eq True text eq True
```

The core algorithm is correct — I verified that `encode(input)` reproduces exactly the
segmentation training ends with (`train-final == encode(input)` → `True`), which is the
subtle part most implementations get wrong. `replace_pairs` is also correctly
non-overlapping/left-to-right.

However, review found defects. Goal: make `train` robust and re-callable, without
changing the core algorithm or its design.

## Findings

### Real bugs

1. **Unguarded `frequencies[0]` → `IndexError`** (`train`)
   Crashes on empty/1-byte input, and when `num_merges` exceeds the pairs available
   (e.g. `train("ab", 5)`). Reproduced for `""`, `"a"`, `train("ab", 5)`.

2. **Retraining corrupts the tokenizer** (`train`)
   `self.merges` is never reset, and `new_token = 256 + merge_idx` restarts at 256, so
   token ids are redefined/collide. Brute force: **3027/4000 round-trip failures** after
   calling `train()` twice.

### Out of scope (confirmed by user)

- **Word boundaries** — training runs on the raw byte stream with no pre-tokenization
  and no word-end marker, so merges cross whitespace (e.g. `"the" + " "`,
  `"the " + "cat"`). Left as-is: a valid byte-level variant that round-trips.
- **`from_bytes` strictness** — `errors="replace"` stays; not changing it.
- Polish items (renaming the `input` param, type annotations, `itertools.pairwise`)
  are not being made.

## Approach

Two small, behavior-preserving fixes to `train`, plus a loud-failure `assert` in
`__main__` so regressions fail instead of printing. Do not restructure the algorithm:
`count_freq` / `replace_pairs` / sequential-replay `encode` / reverse-expansion
`decode` stay as-is (verified correct).

Chosen retrain semantics: **reset** (`self.merges = []`), i.e. each `train()` call
produces a fresh vocab. This is the semantics `encode` already assumes, since it replays
`self.merges` in order.

## Files to modify

- `src/handwritten/bpe.py` — the only file changed.

Not touched: `src/generated/**`, `Makefile`, `pyproject.toml`, `GAPS.md`, `PLAN.md`.

## Reuse

No new helpers needed. Everything operates on the existing functions in the same file
(`count_freq`, `replace_pairs`, `to_bytes`, `from_bytes`) and `full_text` from
`src/generated/common/corpus.py` (already imported).

## Steps

- [ ] **Bug 1** — in `train`, break when there are no pairs:
      `frequencies = count_freq(bytelist); if not frequencies: break`.
- [ ] **Bug 2** — reset the vocab at the start of `train`: `self.merges = []`, so ids
      can never collide on a second call.
- [ ] **Loud failure** — convert the `__main__` prints into `assert`s
      (`decoded == input_bytes`, `from_bytes(decoded) == input_text`), keeping the
      summary print. Reuse the already-computed `input_text` for `input_bytes` so
      `full_text()` is called once.

## Verification

- `uv run src/handwritten/bpe.py` → byte-identical summary line
  (`input 4629 encoded 3611 decoded 4629`), asserts pass, exit 0, no traceback.
- Edge cases now exit cleanly instead of raising `IndexError`:
  `train("", 3)`, `train("a", 3)`, `train("ab", 5)`.
- Retrain check: `train(c1, 3)` then `train(c2, 3)` round-trips for many random pairs
  (previously ~76% failure) — re-run the brute-force script used in review.
- `uv run ruff format --check src/handwritten/bpe.py` stays clean; the only plain
  `ruff check` finding is the pre-existing `RUF007`, which `make fmt` does not enforce.

## Decisions

1. **Scope** — fix bugs 1 and 2, add the loud-failure `assert`; no other polish.
2. **Retrain semantics** — reset the vocab on each `train()` call.
3. **Word boundaries** — out of scope; keep raw-byte training as-is.
4. **Strict `from_bytes`** — out of scope; leave `errors="replace"`.
