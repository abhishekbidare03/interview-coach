"""Phase 3 verification — does a 3B model actually grade answers usefully?

plan.md §5 rates "3B evaluation quality is mediocre" as the highest-impact risk
in the project, and §4 Phase 3 says to find out on a fixture rather than
mid-interview. This is that fixture.

Ten questions across all three modes. For each, three answers written to be
unambiguously strong, partial, and poor. The grader should rank them in that
order. Ranking is what matters — an evaluator that is systematically harsh is
usable, one that cannot tell a good answer from a bad one is not.

The fixtures are hand-written and inline on purpose: they must not change when
the generated bank changes, or the numbers stop being comparable between runs.

Run:  .venv/Scripts/python.exe bench/phase3_eval_test.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from coach.evaluate import Evaluator, elaboration_prompt   # noqa: E402
from coach.llm import LLM                                  # noqa: E402
from coach.schema import Mode, Question, Verdict           # noqa: E402


def Q(topic, mode, text, ref, points) -> Question:
    return Question(topic=topic, difficulty=3, mode=mode, text=text,
                    reference_answer=ref, expected_points=points,
                    follow_up_seeds=["Can you say more about that?"])


# --------------------------------------------------------------------------- #
# Fixtures: (question, strong answer, partial answer, poor answer)
# --------------------------------------------------------------------------- #

FIXTURES: list[tuple[Question, str, str, str]] = [
    (
        Q("os", Mode.FACTUAL,
          "What is the difference between a mutex and a semaphore?",
          "A mutex enforces mutual exclusion so only one thread holds it at a "
          "time, and the thread that acquired it must be the one to release it. "
          "A semaphore is a counter that allows up to N holders, and any thread "
          "can signal it. That ownership property is why a mutex can support "
          "priority inheritance and a semaphore cannot.",
          ["a mutex allows only one thread at a time",
           "the thread that acquired a mutex must be the one to release it",
           "a semaphore is a counter that can allow several threads through",
           "any thread can signal a semaphore, ownership is not tracked"]),
        "A mutex is a locking primitive for mutual exclusion, so exactly one "
        "thread holds it at a time, and critically the thread that acquired it "
        "has to be the one that releases it. A semaphore is really a counter "
        "with atomic increment and decrement, so it can let several threads "
        "through at once, and any thread can signal it since it does not track "
        "ownership.",
        "A mutex lets only one thread into the critical section at a time. A "
        "semaphore is a counter, so it can let more than one through.",
        "They are both used for locking in threads. You use whichever one your "
        "language gives you, they do pretty much the same job.",
    ),
    (
        Q("os", Mode.FACTUAL,
          "What conditions must hold for a deadlock to occur?",
          "Four conditions must all hold: mutual exclusion, hold and wait, no "
          "preemption, and circular wait. Breaking any one of them prevents "
          "deadlock, which is why lock ordering works.",
          ["mutual exclusion, resources cannot be shared",
           "hold and wait, a thread holds one resource while waiting for another",
           "no preemption, resources cannot be forcibly taken away",
           "circular wait, a cycle exists in the wait-for graph"]),
        "All four Coffman conditions have to hold at once. Mutual exclusion, so "
        "the resource cannot be shared. Hold and wait, where a thread keeps one "
        "lock while waiting on another. No preemption, so you cannot forcibly "
        "take a lock back. And circular wait, a cycle in the wait-for graph. "
        "Break any one and you cannot deadlock, which is why consistent lock "
        "ordering works.",
        "You need a circular wait, where thread A holds a lock that B wants and "
        "B holds one that A wants. And the resources have to be exclusive.",
        "A deadlock is when the program freezes because two threads are stuck. "
        "You fix it by adding a timeout.",
    ),
    (
        Q("dbms", Mode.FACTUAL,
          "Why might adding an index make your database slower?",
          "Indexes have to be updated on every insert, update, and delete, so "
          "write throughput drops. They also consume disk and memory, and a "
          "low-selectivity index may be ignored by the planner in favour of a "
          "sequential scan, so you pay the write cost for no read benefit.",
          ["every write must also update the index, slowing inserts and updates",
           "indexes consume additional disk and memory",
           "a low-selectivity index may not be used by the query planner",
           "a sequential scan can beat an index scan on small or low-selectivity data"]),
        "Because every insert, update and delete now has to maintain the index "
        "as well as the table, so writes get slower. The index also costs disk "
        "and memory. And if the column has low selectivity, the planner will "
        "often ignore the index and do a sequential scan anyway, so you pay the "
        "write penalty and get nothing back on reads.",
        "Indexes slow down writes because the database has to keep them up to "
        "date whenever you insert a row.",
        "Indexes always make things faster, that is the whole point of them. "
        "Maybe if you have too many it uses more space.",
    ),
    (
        Q("networking", Mode.FACTUAL,
          "What does it mean for an HTTP method to be idempotent, and why does it matter?",
          "An idempotent method produces the same server state whether it is "
          "called once or many times. GET, PUT and DELETE are idempotent; POST "
          "is not. It matters because it lets clients, proxies and load "
          "balancers safely retry a request after a timeout without risking a "
          "duplicate side effect.",
          ["calling it many times leaves the same server state as calling it once",
           "GET, PUT and DELETE are idempotent while POST is not",
           "it makes retrying a failed or timed-out request safe",
           "it is about the resulting state, not about the response being identical"]),
        "It means making the same call repeatedly leaves the server in the same "
        "state as making it once. GET, PUT and DELETE are idempotent, POST is "
        "not. It matters because if a request times out you often cannot tell "
        "whether it landed, so being idempotent lets the client or a proxy retry "
        "safely. Note it is about the resulting state, not about getting a byte "
        "identical response back.",
        "It means you can call it more than once and nothing bad happens. GET is "
        "idempotent, POST is not.",
        "It means the request always returns exactly the same response body "
        "every time you call it.",
    ),
    (
        Q("python", Mode.FACTUAL,
          "What is the GIL and what are its practical consequences?",
          "The global interpreter lock is a mutex in CPython that allows only "
          "one thread to execute Python bytecode at a time. So threads do not "
          "give you parallel speedup on CPU-bound work, but they are still "
          "useful for I/O-bound work because the lock is released during I/O. "
          "For CPU parallelism you use multiprocessing or a native extension.",
          ["only one thread executes Python bytecode at a time",
           "threads give no parallel speedup for CPU-bound work",
           "threads still help for I/O-bound work because the lock is released during I/O",
           "multiprocessing or native extensions are the route to CPU parallelism"]),
        "The global interpreter lock is a mutex in CPython that means only one "
        "thread runs Python bytecode at any moment. The practical consequence is "
        "that threading buys you nothing for CPU-bound work. It still helps for "
        "I/O-bound work, because the lock gets released while waiting on I/O. If "
        "you genuinely need CPU parallelism you reach for multiprocessing, or a "
        "C extension that drops the lock.",
        "It is a lock that stops more than one thread running at once in Python, "
        "so threading does not really make CPU work faster.",
        "The GIL is Python's garbage collector. It cleans up objects you are no "
        "longer using so you do not have to manage memory.",
    ),
    (
        Q("dsa", Mode.FACTUAL,
          "When would you use a hash table over a balanced binary search tree?",
          "A hash table gives average constant time lookup versus logarithmic "
          "for a tree, so it wins when you only need key lookup. A tree keeps "
          "keys in sorted order, so it wins when you need range queries, ordered "
          "traversal, or a predecessor lookup. Hash tables also have worst case "
          "linear behaviour on collisions and no ordering guarantee.",
          ["hash tables give average constant time lookup versus logarithmic for a tree",
           "trees keep keys in sorted order, enabling range queries and ordered traversal",
           "hash tables have worst case linear lookup under heavy collisions",
           "a hash table gives no ordering guarantee at all"]),
        "If all I need is point lookup by key, a hash table wins because it is "
        "average constant time against log n for the tree. I would pick the "
        "balanced tree the moment I need ordering, so range queries, in-order "
        "traversal, or finding the nearest key below something. The catch with "
        "hashing is that worst case it degrades to linear if collisions pile up, "
        "and it gives you no ordering at all.",
        "Hash tables are O(1) for lookup and trees are O(log n), so hash tables "
        "are faster for looking things up.",
        "You should always use a hash table, they are the fastest data structure "
        "for everything and trees are mostly obsolete now.",
    ),
    (
        Q("system_design", Mode.OPEN_ENDED,
          "How would you design a URL shortener?",
          "You need an ID generation strategy, a key-value store mapping short "
          "code to long URL, and a redirect service. Reads vastly outnumber "
          "writes so the read path should be cached aggressively. Consider "
          "collision handling, whether codes are guessable, custom aliases, "
          "expiry, and analytics on redirects.",
          ["a mapping from a short code to the long URL in a key-value store",
           "a strategy for generating short codes, such as a counter with base62 or a hash",
           "reads vastly outnumber writes so the read path should be cached",
           "handling collisions or guaranteeing uniqueness of generated codes",
           "the redirect itself and which HTTP status code to use",
           "scaling considerations such as sharding or read replicas"]),
        "At the core it is a key-value mapping from a short code to the long "
        "URL, so I would reach for something like Redis in front of a durable "
        "store. For code generation I would use a monotonic counter encoded in "
        "base62 rather than hashing, because that guarantees uniqueness without "
        "collision handling, though it does make codes guessable so I would add "
        "a salt if that matters. The traffic pattern is extremely read heavy, "
        "maybe a hundred to one, so the redirect path should be cache first and "
        "I would push it to a CDN. The redirect itself should be a 301 if "
        "permanent, but a 302 if I want the analytics on every hit. To scale, "
        "shard by short code and add read replicas.",
        "I would store the short code and the long URL in a database table and "
        "look it up on each request, then redirect the user. I would generate "
        "the code with a random string and check if it already exists.",
        "I would make a website where you paste a URL and it gives you a short "
        "one back. You would need a database to save them.",
    ),
    (
        Q("behavioural", Mode.BEHAVIOURAL,
          "Tell me about a time you disagreed with a technical decision on your team.",
          "A strong answer names a specific decision and context, explains the "
          "candidate's own reasoning and what they personally did to raise it, "
          "states how it actually resolved including if they were overruled, and "
          "reflects honestly on what they would do differently.",
          ["names a specific decision with enough context to follow",
           "says what they personally did, not what the team did",
           "gives a concrete outcome including whether they were overruled",
           "reflects on what they learned or would do differently"]),
        "We were about to move our background jobs onto a message queue, and I "
        "disagreed because our volume was about two hundred jobs a day and a "
        "cron table would have done it. I wrote up a one page comparison with "
        "the operational cost of running the broker and took it to the tech lead "
        "in our weekly sync. He heard me out but went with the queue anyway, "
        "because he was planning for a product launch I did not know about. It "
        "shipped and was fine. What I took from it was that I had argued purely "
        "from current load, and I should have asked what the roadmap looked like "
        "before writing the doc at all.",
        "I disagreed with my team about using a message queue for our background "
        "jobs. I thought it was overkill for our scale. We discussed it and "
        "ended up going with the queue. It worked out okay in the end.",
        "Yeah, this happens a lot actually. Usually when people pick the wrong "
        "technology you just have to go along with it because that is how teams "
        "work. You pick your battles.",
    ),
    (
        Q("behavioural", Mode.BEHAVIOURAL,
          "Tell me about a mistake you made that affected other people.",
          "A strong answer owns a real mistake without deflecting, describes the "
          "concrete impact, explains what the candidate personally did to fix it "
          "and to communicate it, and identifies the systemic change that "
          "prevents a recurrence.",
          ["describes a real mistake they actually made, without deflecting blame",
           "states the concrete impact on other people",
           "says what they personally did to fix it and to communicate it",
           "identifies a change that prevents it happening again"]),
        "I ran a migration against production that dropped a column I thought "
        "was unused. It was still being read by our billing export, so the "
        "nightly invoice run failed and about forty customers did not get "
        "invoiced that night. I noticed the alert within the hour, restored the "
        "column from the backup and backfilled from the write-ahead log, then "
        "posted in the incident channel and messaged the finance lead directly "
        "rather than waiting for standup. The real fix was that we had no way to "
        "know who read a column, so I added a check to our migration template "
        "that greps the codebase for the column name and fails the migration if "
        "it finds a reference.",
        "I once deleted a database column that turned out to still be in use and "
        "it broke a report. I put it back once we noticed. We were more careful "
        "with migrations after that.",
        "Honestly the process was pretty broken there, there was no proper "
        "staging environment so things broke all the time. Everyone made "
        "mistakes, it was that kind of place.",
    ),
    (
        Q("behavioural", Mode.BEHAVIOURAL,
          "Tell me about a time you had to learn something quickly.",
          "A strong answer names the specific thing, why the timeline was tight, "
          "the concrete method used to get up to speed, the result, and a "
          "reflection on the learning approach itself.",
          ["names the specific thing they had to learn and why it was urgent",
           "describes the concrete method they used to learn it",
           "gives a concrete result or outcome",
           "reflects on what they would do differently next time"]),
        "Our only Kubernetes person left two weeks before a launch and I had "
        "never touched it. I gave myself three days, and rather than reading the "
        "docs front to back I took our existing manifests and deliberately broke "
        "them one at a time in a scratch cluster to see what each field actually "
        "did. By the launch I could do rollouts and read the events well enough "
        "to debug a crashloop. I would do the breaking-things approach again, "
        "but I would have written down what I learned as I went, because I lost "
        "a chunk of it over the following month.",
        "I had to learn Kubernetes quickly when a colleague left before a "
        "launch. I read the documentation and watched some tutorials, and I got "
        "up to speed well enough to handle the deployment.",
        "I learn new things all the time, I would say I am a pretty fast "
        "learner. Every job has new technology so you just pick it up as you go "
        "along really.",
    ),
]

TIERS = ("strong", "partial", "poor")
RANK = {Verdict.CORRECT: 2, Verdict.STRONG: 2,
        Verdict.PARTIAL: 1, Verdict.ADEQUATE: 1,
        Verdict.INCORRECT: 0, Verdict.WEAK: 0, Verdict.UNCLEAR: 0}


async def main() -> int:
    async with LLM() as llm:
        if not (await llm.health())["has_model"]:
            print("ollama is missing the configured model", file=sys.stderr)
            return 1

        ev = Evaluator(llm, seed=0)
        rows, mono, ranks_ok, times = [], 0, 0, []

        print(f"grading {len(FIXTURES)} questions x 3 answers = "
              f"{len(FIXTURES) * 3} gradings\n")

        for q, *answers in FIXTURES:
            results = []
            for answer in answers:
                r = await ev.evaluate(q, answer)
                results.append(r)
                times.append(r.grade_ms)

            scores = [r.score for r in results]
            ranked = [RANK[r.verdict] for r in results]
            # Strictly decreasing score is the strong claim; correct verdict
            # ordering (ties allowed) is the one that actually matters in use.
            is_mono = scores[0] > scores[1] > scores[2]
            is_ranked = ranked[0] >= ranked[1] >= ranked[2] and ranked[0] > ranked[2]
            mono += is_mono
            ranks_ok += is_ranked

            print(f"{'OK ' if is_ranked else '>>>'} [{q.mode:<11}] {q.text[:56]}")
            for tier, r in zip(TIERS, results):
                print(f"      {tier:<8} {str(r.verdict):<10} "
                      f"score {r.score:.2f}  "
                      f"covered {len(r.covered)}/{len(q.expected_points)}  "
                      f"{r.grade_ms:.0f}ms"
                      f"{'  follow-up' if r.should_follow_up else ''}")
            if not is_ranked:
                print(f"      ^ ordering wrong: {[str(r.verdict) for r in results]}")
            rows.append({"question": q.text, "mode": str(q.mode),
                         "verdicts": [str(r.verdict) for r in results],
                         "scores": scores, "ranked_correctly": is_ranked})
            print()

        n = len(FIXTURES)
        section = "=" * 72
        print(section)
        print("PHASE 3 EXIT CRITERIA")
        print(section)

        # Per-mode breakdown: behavioural grading is the one most likely to be
        # weak, and an aggregate number would hide that.
        by_mode: dict[str, list[bool]] = {}
        for row in rows:
            by_mode.setdefault(row["mode"], []).append(row["ranked_correctly"])
        for mode, oks in by_mode.items():
            print(f"    {mode:<12} {sum(oks)}/{len(oks)} ranked correctly")

        checks = {
            "verdicts ranked strong >= partial >= poor":
                (ranks_ok >= n - 1, f"{ranks_ok}/{n}"),
            "scores strictly decreasing (stronger claim)":
                (mono >= n * 0.7, f"{mono}/{n}"),
            "median grading latency < 900 ms":
                (statistics.median(times) < 900,
                 f"{statistics.median(times):.0f} ms"),
        }
        print()
        for label, (good, val) in checks.items():
            print(f"  [{'PASS' if good else 'FAIL'}] {label:48s} {val}")

        # Show one full spoken response so the feedback quality is inspectable,
        # not just the numbers.
        q, _, partial, _ = FIXTURES[0]
        r = await ev.evaluate(q, partial)
        text = await llm.complete(elaboration_prompt(q, partial, r))
        print(f"\n{section}\nSAMPLE SPOKEN FEEDBACK (partial answer)\n{section}")
        print(f"  Q: {q.text}")
        print(f"  verdict: {r.verdict}  missing: {r.missing}")
        print(f"  spoken: {r.opening_line} {text.strip()}")

        (ROOT / "bench" / "phase3_results.json").write_text(json.dumps(
            {"rows": rows, "checks": {k: v[0] for k, v in checks.items()},
             "median_grade_ms": statistics.median(times)}, indent=2))
        return 0 if all(v[0] for v in checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
