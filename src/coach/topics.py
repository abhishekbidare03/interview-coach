"""Topic taxonomy.

Each topic carries three things the rest of the system needs:

* **subtopics** — the diversity lever. Asking a 3B model for "40 operating
  systems questions" yields the same eight questions with different wording.
  Asking it for "one question about priority inversion, difficulty 4" does not.
  This is the concrete mechanism behind plan.md §2.2.
* **vocabulary** — technical terms fed to Whisper as `initial_prompt`
  (plan.md §2.4.1). Without it `small.en` normalises jargon into common English
  and the evaluator grades a misheard answer, which is the worst failure the
  system has.
* **mode** — how answers here must be judged (plan.md §2.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Mode


@dataclass(frozen=True)
class Topic:
    key: str
    label: str
    mode: Mode
    subtopics: tuple[str, ...]
    vocabulary: tuple[str, ...] = ()
    blurb: str = ""
    needs_resume: bool = False

    @property
    def vocab_prompt(self) -> str:
        return ", ".join(self.vocabulary)


TOPICS: dict[str, Topic] = {}


def _add(t: Topic) -> Topic:
    TOPICS[t.key] = t
    return t


# --------------------------------------------------------------------------- #
# Technical — factual
# --------------------------------------------------------------------------- #

_add(Topic(
    key="dsa", label="Data Structures & Algorithms", mode=Mode.FACTUAL,
    blurb="Complexity, core structures, and the reasoning behind algorithm choice.",
    subtopics=(
        "time and space complexity analysis", "arrays versus linked lists",
        "hash tables and collision handling", "binary search and its variants",
        "sorting algorithm tradeoffs", "binary trees and traversals",
        "balanced trees and when they matter", "heaps and priority queues",
        "graph representations", "breadth-first versus depth-first search",
        "shortest path algorithms", "dynamic programming versus greedy",
        "recursion and stack depth", "two pointers and sliding window",
        "tries and prefix matching", "union-find",
        "amortised analysis", "in-place algorithms and space tradeoffs",
    ),
    vocabulary=(
        "big O notation", "amortised", "hash table", "collision", "chaining",
        "open addressing", "binary search tree", "AVL", "red-black tree",
        "heap", "trie", "adjacency list", "Dijkstra", "memoization",
        "dynamic programming", "quicksort", "mergesort", "union-find",
    ),
))

_add(Topic(
    key="os", label="Operating Systems", mode=Mode.FACTUAL,
    blurb="Processes, concurrency, memory, and scheduling.",
    subtopics=(
        "processes versus threads", "context switching costs",
        "mutexes versus semaphores", "deadlock conditions and prevention",
        "race conditions and atomicity", "virtual memory and paging",
        "page replacement policies", "thrashing", "CPU scheduling algorithms",
        "priority inversion", "user space versus kernel space", "system calls",
        "interrupts", "memory fragmentation", "copy-on-write",
        "inter-process communication", "file descriptors", "zombie and orphan processes",
    ),
    vocabulary=(
        "mutex", "semaphore", "deadlock", "livelock", "race condition",
        "context switch", "virtual memory", "paging", "page fault", "TLB",
        "thrashing", "preemptive", "round robin", "priority inversion",
        "kernel space", "system call", "copy-on-write", "IPC", "spinlock",
    ),
))

_add(Topic(
    key="dbms", label="Databases", mode=Mode.FACTUAL,
    blurb="Relational modelling, transactions, indexing, and query performance.",
    subtopics=(
        "normalisation and its tradeoffs", "primary and foreign keys",
        "ACID properties", "isolation levels and anomalies",
        "how B-tree indexes work", "when an index hurts", "composite index ordering",
        "query execution plans", "joins and join algorithms",
        "transactions and rollback", "optimistic versus pessimistic locking",
        "SQL versus NoSQL tradeoffs", "sharding and partitioning",
        "replication and read replicas", "the N+1 query problem",
        "denormalisation for read performance", "connection pooling",
    ),
    vocabulary=(
        "normalisation", "denormalisation", "ACID", "atomicity", "isolation level",
        "read committed", "repeatable read", "serializable", "phantom read",
        "B-tree", "composite index", "query plan", "hash join", "nested loop join",
        "sharding", "replication", "PostgreSQL", "MySQL", "MongoDB", "Redis",
    ),
))

_add(Topic(
    key="networking", label="Networking", mode=Mode.FACTUAL,
    blurb="Protocols, the request lifecycle, and what breaks between machines.",
    subtopics=(
        "TCP versus UDP", "the TCP handshake", "what happens when you type a URL",
        "DNS resolution", "HTTP methods and idempotency", "HTTP status codes",
        "HTTPS and TLS", "cookies versus tokens", "CORS",
        "REST versus RPC", "WebSockets versus polling", "load balancing",
        "caching layers and cache headers", "CDNs", "latency versus bandwidth",
        "network partitions", "rate limiting",
    ),
    vocabulary=(
        "TCP", "UDP", "handshake", "DNS", "HTTP", "HTTPS", "TLS", "idempotent",
        "CORS", "REST", "WebSocket", "load balancer", "CDN", "latency",
        "bandwidth", "packet loss", "rate limiting", "reverse proxy", "nginx",
    ),
))

_add(Topic(
    key="python", label="Python", mode=Mode.FACTUAL,
    blurb="Language semantics, the data model, and idiomatic use.",
    # Ordered roughly by depth, because the Python bank is hand-written
    # (scripts/build_python_bank.py) and deliberately bottom-heavy: a screening
    # round spends most of its time checking that the basics are solid.
    subtopics=(
        # basics
        "list versus tuple", "dictionaries", "sets", "identity versus equality",
        "mutability of strings", "dynamic typing", "slicing", "None",
        "list comprehensions", "loops", "modules and imports", "docstrings",
        # everyday
        "shallow versus deep copy", "args and kwargs", "decorators",
        "generators", "method types", "default mutable arguments",
        "context managers", "iterators versus iterables", "scope",
        "sort versus sorted", "lambda", "f-strings", "truthiness",
        # depth
        "the GIL", "threading versus multiprocessing", "yield",
        "dunder methods", "closures", "duck typing", "method resolution order",
        "composition versus inheritance", "garbage collection",
        "attribute lookup", "equality and hashing",
        # stretch
        "asyncio", "slots", "reference cycles", "metaclasses", "descriptors",
        "profiling", "parallelism in practice", "import system",
    ),
    vocabulary=(
        "GIL", "global interpreter lock", "generator", "yield", "decorator",
        "context manager", "comprehension", "kwargs", "dunder", "iterable",
        "iterator", "closure", "asyncio", "coroutine", "multiprocessing",
        "reference counting", "garbage collection", "duck typing", "mutable",
        # Terms the hand-written bank actually asks about. Whisper normalises
        # these into ordinary English without the bias prompt — "dunder" alone
        # comes back as "under" often enough to break grading.
        "immutable", "tuple", "hashable", "slicing", "lambda", "f-string",
        "enumerate", "metaclass", "descriptor", "slots", "monkey patching",
        "method resolution order", "shallow copy", "deep copy", "pickling",
    ),
))

# --------------------------------------------------------------------------- #
# Technical — open-ended
# --------------------------------------------------------------------------- #

_add(Topic(
    key="system_design", label="System Design", mode=Mode.OPEN_ENDED,
    blurb="Architecture, tradeoffs, and how systems fail at scale.",
    subtopics=(
        "design a URL shortener", "design a rate limiter",
        "design a news feed", "design a chat application",
        "design a file storage service", "design a ride-hailing dispatch system",
        "design a notification service", "design a search autocomplete",
        "caching strategy and invalidation", "database choice for a given workload",
        "handling traffic spikes", "designing for failure and retries",
        "consistency versus availability tradeoffs", "API design and versioning",
        "queueing and asynchronous processing", "observability and alerting",
    ),
    vocabulary=(
        "horizontal scaling", "vertical scaling", "load balancer", "sharding",
        "consistent hashing", "cache invalidation", "eventual consistency",
        "CAP theorem", "message queue", "Kafka", "idempotency", "throughput",
        "p99 latency", "circuit breaker", "backpressure", "microservices",
    ),
))

# --------------------------------------------------------------------------- #
# Behavioural
# --------------------------------------------------------------------------- #

_add(Topic(
    key="behavioural", label="Behavioural", mode=Mode.BEHAVIOURAL,
    blurb="Past experience, judged on structure and specificity — not correctness.",
    subtopics=(
        "a conflict with a teammate", "a project that failed",
        "a time you received hard feedback", "a deadline you missed",
        "convincing someone who disagreed with you",
        "a time you had to learn something quickly",
        "a mistake you made in production", "working with an unclear requirement",
        "prioritising when everything was urgent", "a time you disagreed with a decision",
        "helping someone who was struggling", "taking ownership of something unglamorous",
        "a technical decision you later regretted", "handling an interruption mid-project",
        "your proudest piece of work", "a time you asked for help",
    ),
    vocabulary=(
        "stakeholder", "deadline", "sprint", "retrospective", "code review",
        "escalate", "scope", "requirement", "on-call", "postmortem",
    ),
))

_add(Topic(
    key="resume", label="Résumé & Projects", mode=Mode.BEHAVIOURAL,
    blurb="Questions about your actual projects. Requires your CV.",
    needs_resume=True,
    subtopics=(
        "why you chose that technology", "the hardest bug in that project",
        "what you would rebuild differently", "how you tested it",
        "what you personally owned versus the team", "how it performed under load",
        "what you learned from it", "what you would add next",
    ),
    vocabulary=(),
))


# --------------------------------------------------------------------------- #

TECHNICAL = ("dsa", "os", "dbms", "networking", "python", "system_design")
BEHAVIOURAL = ("behavioural",)

# Weights for the "Random Interview" button (plan.md R3). Roughly 60/40
# technical to behavioural, which is close to a real screening loop.
RANDOM_MIX: dict[str, float] = {
    "dsa": 0.16, "os": 0.13, "dbms": 0.13, "networking": 0.10,
    "python": 0.10, "system_design": 0.13,
    "behavioural": 0.25,
}


def get(key: str) -> Topic:
    if key not in TOPICS:
        raise KeyError(f"unknown topic {key!r}; known: {sorted(TOPICS)}")
    return TOPICS[key]


def vocabulary_for(keys: list[str]) -> str:
    """Combined STT bias prompt for a set of topics, deduped and length-capped.

    Whisper's `initial_prompt` competes with the audio for the model's attention,
    so an unbounded term dump degrades transcription instead of helping it.
    """
    seen: list[str] = []
    for k in keys:
        for term in TOPICS[k].vocabulary:
            if term not in seen:
                seen.append(term)
    return ", ".join(seen[:60])


def selectable() -> list[dict]:
    """Topic list for the UI picker."""
    return [
        {"key": t.key, "label": t.label, "mode": str(t.mode),
         "blurb": t.blurb, "needs_resume": t.needs_resume,
         "subtopics": len(t.subtopics)}
        for t in TOPICS.values()
    ]
