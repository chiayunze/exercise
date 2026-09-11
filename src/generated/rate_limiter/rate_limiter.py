"""Rate limiting: five classic algorithms behind one interface.

Interview talking points:
- The job: given a key (user, IP, API key), decide whether each request is
  allowed. All five algorithms below answer `allow(key) -> bool`; they differ
  only in how they count.
- Token bucket: a bucket of `capacity` tokens refills at `rate` tokens/sec.
  Each request takes one token. Allows bursts up to `capacity`, then admits at
  the steady refill rate. The friendliest for real traffic (AWS/Stripe style).
- Leaky bucket: the mirror image. Requests add to a bucket that drains at a
  constant rate; a request is dropped when the bucket is full. This *smooths*
  output to a fixed rate instead of permitting bursts. (We implement the
  meter/drop variant; the queue variant would delay requests instead.)
- Fixed window counter: one counter per aligned time window, reset at the
  boundary. O(1) memory and trivially simple, but a burst straddling the
  boundary can admit up to 2x `limit` in a rolling window.
- Sliding window log: keep the timestamp of every admitted request; evict
  those older than the window. Exact, but O(limit) memory per key.
- Sliding window counter: keep only two aligned-window counts and weight the
  previous one by how much of it still overlaps the trailing window. O(1)
  memory approximation of the log, with a small error near boundaries.
- Refill/eviction is computed lazily from the clock on each request -- no
  background thread, no timers.
- We take the clock as a parameter so behaviour is deterministic and testable;
  production code would use a monotonic clock. This reference is
  single-threaded: a real implementation needs a per-key lock (or an atomic
  store such as Redis) because read-modify-write races lose counts.

Run:  uv run src/generated/rate_limiter/rate_limiter.py
"""

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class RateLimiter:
    """Base class: per-key state plus a shared clock.

    Subclasses implement `_new_state` (initial state for a fresh key) and
    `_decide` (mutate the state, return whether the request is allowed).
    """

    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self._now = now or time.monotonic
        self._states: dict[str, Any] = {}

    def allow(self, key: str = "default") -> bool:
        """Record a request for `key`; return True if it is allowed."""
        state = self._states.get(key)
        if state is None:
            state = self._new_state()
            self._states[key] = state
        return self._decide(state)

    def _new_state(self) -> Any:
        raise NotImplementedError

    def _decide(self, state: Any) -> bool:
        raise NotImplementedError


# --------------------------------------------------------------- token bucket


@dataclass
class _TokenBucket:
    tokens: float
    last: float


class TokenBucket(RateLimiter):
    """Burst up to `capacity`, then admit at `refill_rate` requests/sec."""

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        now: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(now)
        self.capacity = capacity
        self.refill_rate = refill_rate

    def _new_state(self) -> _TokenBucket:
        return _TokenBucket(tokens=self.capacity, last=self._now())

    def _decide(self, state: _TokenBucket) -> bool:
        now = self._now()
        elapsed = max(0.0, now - state.last)
        state.tokens = min(self.capacity, state.tokens + elapsed * self.refill_rate)
        state.last = now
        if state.tokens >= 1.0:
            state.tokens -= 1.0
            return True
        return False


# --------------------------------------------------------------- leaky bucket


@dataclass
class _LeakyBucket:
    level: float
    last: float


class LeakyBucket(RateLimiter):
    """Smooth output: drain a full bucket at `leak_rate` requests/sec.

    Requests "pour" 1 unit into the bucket; once the bucket would overflow
    past `capacity`, further requests are dropped until it drains enough.
    """

    def __init__(
        self,
        capacity: float,
        leak_rate: float,
        now: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(now)
        self.capacity = capacity
        self.leak_rate = leak_rate

    def _new_state(self) -> _LeakyBucket:
        return _LeakyBucket(level=0.0, last=self._now())

    def _decide(self, state: _LeakyBucket) -> bool:
        now = self._now()
        elapsed = max(0.0, now - state.last)
        state.level = max(0.0, state.level - elapsed * self.leak_rate)
        state.last = now
        if state.level + 1.0 <= self.capacity:
            state.level += 1.0
            return True
        return False


# ---------------------------------------------------- fixed window counter


@dataclass
class _FixedWindow:
    window_start: float
    count: int


class FixedWindowCounter(RateLimiter):
    """`limit` requests per aligned `window`-second window."""

    def __init__(
        self,
        limit: int,
        window: float = 1.0,
        now: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(now)
        self.limit = limit
        self.window = window

    def _new_state(self) -> _FixedWindow:
        return _FixedWindow(window_start=self._window_start(), count=0)

    def _window_start(self) -> float:
        return math.floor(self._now() / self.window) * self.window

    def _decide(self, state: _FixedWindow) -> bool:
        start = self._window_start()
        if start != state.window_start:
            state.window_start = start
            state.count = 0
        if state.count < self.limit:
            state.count += 1
            return True
        return False


# ---------------------------------------------------------- sliding window log


@dataclass
class _SlidingLog:
    log: deque[float] = field(default_factory=deque)


class SlidingWindowLog(RateLimiter):
    """Exact `limit` requests per rolling `window` seconds.

    Stores every admitted timestamp; O(limit) memory per key.
    """

    def __init__(
        self,
        limit: int,
        window: float = 1.0,
        now: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(now)
        self.limit = limit
        self.window = window

    def _new_state(self) -> _SlidingLog:
        return _SlidingLog()

    def _decide(self, state: _SlidingLog) -> bool:
        now = self._now()
        cutoff = now - self.window
        while state.log and state.log[0] <= cutoff:
            state.log.popleft()
        if len(state.log) < self.limit:
            state.log.append(now)
            return True
        return False


# ------------------------------------------------------ sliding window counter


@dataclass
class _SlidingCounter:
    window_index: int
    current: int
    previous: int


class SlidingWindowCounter(RateLimiter):
    """O(1)-memory approximation of the sliding log.

    Estimate the count in the trailing window as
    `previous * overlap_fraction + current`, where `overlap_fraction` is the
    share of the previous aligned window still inside the rolling window.
    """

    def __init__(
        self,
        limit: int,
        window: float = 1.0,
        now: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(now)
        self.limit = limit
        self.window = window

    def _new_state(self) -> _SlidingCounter:
        return _SlidingCounter(
            window_index=int(self._now() // self.window), current=0, previous=0
        )

    def _decide(self, state: _SlidingCounter) -> bool:
        now = self._now()
        index = int(now // self.window)
        if index != state.window_index:
            state.previous = state.current if index == state.window_index + 1 else 0
            state.current = 0
            state.window_index = index
        elapsed = (now - index * self.window) / self.window
        estimate = state.previous * (1.0 - elapsed) + state.current
        if estimate < self.limit:
            state.current += 1
            return True
        return False


# ------------------------------------------------------------------- demo


class FakeClock:
    """Deterministic clock: `advance` moves it, calling it reads it."""

    def __init__(self, start: float = 0.0) -> None:
        self.time = start

    def __call__(self) -> float:
        return self.time

    def advance(self, dt: float) -> None:
        self.time += dt


def demo() -> None:
    print("== Steady traffic: 1 request / 0.1s for 3s (30 requests) ==")
    times = [i * 0.1 for i in range(30)]

    specs: list[tuple[str, Callable[[FakeClock], RateLimiter]]] = [
        ("token bucket", lambda c: TokenBucket(5, 5.0, c)),
        ("leaky bucket", lambda c: LeakyBucket(5, 5.0, c)),
        ("fixed window", lambda c: FixedWindowCounter(5, 1.0, c)),
        ("sliding log", lambda c: SlidingWindowLog(5, 1.0, c)),
        ("sliding counter", lambda c: SlidingWindowCounter(5, 1.0, c)),
    ]
    for name, make in specs:
        clock = FakeClock()
        limiter = make(clock)
        allowed = 0
        for t in times:
            clock.time = t
            allowed += limiter.allow("client")
        print(f"  {name:<16} allowed {allowed:2d} / 30, blocked {30 - allowed:2d}")

    print("\n== Boundary burst: 3 requests at t=0.9, then 3 at t=1.0 ==")
    print("   (limit 3 per 1s window; fixed windows reset at t=1.0)")
    times = [0.9, 0.9, 0.9, 1.0, 1.0, 1.0]

    clock = FakeClock()
    fixed = FixedWindowCounter(3, 1.0, clock)
    log = SlidingWindowLog(3, 1.0, clock)
    counter = SlidingWindowCounter(3, 1.0, clock)
    totals = {"fixed window": 0, "sliding log": 0, "sliding counter": 0}
    for t in times:
        clock.time = t
        totals["fixed window"] += fixed.allow("client")
        totals["sliding log"] += log.allow("client")
        totals["sliding counter"] += counter.allow("client")
    for name, n in totals.items():
        print(f"  {name:<16} admitted {n} / 6")
    print("   fixed admits 6 (2x limit across the boundary); both sliding")
    print("   variants admit ~3, which is the whole point of sliding.")


if __name__ == "__main__":
    demo()

    # --- token bucket: burst up to capacity, refill over time, then allow ---
    clock = FakeClock()
    tb = TokenBucket(capacity=3, refill_rate=1.0, now=clock)
    assert [tb.allow("k") for _ in range(4)] == [True, True, True, False]
    clock.advance(1.0)
    assert tb.allow("k") is True  # one token refilled
    assert tb.allow("k") is False

    # --- leaky bucket: burst capacity, then drain-limited ---
    clock = FakeClock()
    lb = LeakyBucket(capacity=3, leak_rate=1.0, now=clock)
    assert [lb.allow("k") for _ in range(4)] == [True, True, True, False]
    clock.advance(1.0)
    assert lb.allow("k") is True
    assert lb.allow("k") is False

    # --- fixed window resets exactly at the aligned boundary ---
    clock = FakeClock()
    fw = FixedWindowCounter(limit=2, window=1.0, now=clock)
    assert [fw.allow("k") for _ in range(3)] == [True, True, False]
    clock.time = 1.0
    assert fw.allow("k") is True  # new window, counter reset

    # --- sliding log: exact rolling limit ---
    clock = FakeClock()
    sl = SlidingWindowLog(limit=2, window=1.0, now=clock)
    assert [sl.allow("k") for _ in range(3)] == [True, True, False]
    clock.time = 0.5
    assert sl.allow("k") is False  # still inside the rolling window
    clock.time = 1.0
    assert sl.allow("k") is True  # t=0.0 entry has aged out (<= cutoff)

    # --- sliding counter: same limit, O(1) memory ---
    clock = FakeClock()
    sc = SlidingWindowCounter(limit=2, window=1.0, now=clock)
    assert [sc.allow("k") for _ in range(3)] == [True, True, False]

    # --- per-key isolation: exhausting one key must not affect another ---
    limiter = TokenBucket(capacity=1, refill_rate=0.0, now=FakeClock())
    assert limiter.allow("alice") is True
    assert limiter.allow("alice") is False
    assert limiter.allow("bob") is True

    # --- the classic fixed-vs-sliding boundary difference ---
    clock = FakeClock()
    fixed = FixedWindowCounter(3, 1.0, clock)
    log = SlidingWindowLog(3, 1.0, clock)
    admitted = {"fixed": 0, "log": 0}
    for t in [0.9, 0.9, 0.9, 1.0, 1.0, 1.0]:
        clock.time = t
        admitted["fixed"] += fixed.allow("k")
        admitted["log"] += log.allow("k")
    assert admitted["fixed"] == 6, admitted
    assert admitted["log"] == 3, admitted

    print("\n[rate_limiter] all asserts passed")
