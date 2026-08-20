"""Offline question-bank generator.

plan.md §2.2: a 3B model cannot invent good interview questions in the middle of
a conversation, but it can write decent ones when nobody is waiting. This script
is where that happens — no latency budget, retries are free, and the output is
reviewed before it is ever used.

The diversity mechanism is the important part. Asking for "40 operating systems
questions" produces the same eight questions reworded. Asking for one question
about *priority inversion* at *difficulty 4* does not. So generation is driven
by the (subtopic x difficulty) grid in `topics.py`, one question per cell.

Usage:
    python scripts/generate_bank.py --topic os --topic dsa
    python scripts/generate_bank.py --all --per-subtopic 2
    python scripts/generate_bank.py --topic os --review     # print for review
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from coach import topics as T                      # noqa: E402
from coach.config import BANK                      # noqa: E402
from coach.llm import LLM                          # noqa: E402
from coach.schema import (Mode, Question,          # noqa: E402
                          load_questions, save_questions)

# Difficulty is sampled per subtopic rather than fixed, so a bank has a spread
# to build a curve from (see blueprint.py).
DIFFICULTIES = (2, 3, 4)

SPEC = {
    Mode.FACTUAL: """\
Write ONE technical interview question with a defensible correct answer.

Return JSON with exactly these keys:
  "text"              the question, one or two sentences, asked conversationally
  "reference_answer"  the correct answer in 2-4 sentences, spoken plainly
  "expected_points"   3-5 short strings; the specific things a complete answer covers
  "follow_up_seeds"   2 short probing follow-up questions

Rules:
- The question must be answerable OUT LOUD in under 90 seconds. No code writing.
- No "implement X" or "write a function" questions - this is a spoken interview.
- reference_answer will be read aloud to a candidate who got it wrong, so write
  it as speech: no markdown, no bullet points, no code blocks, no symbols.
- Each expected_point must STATE THE FACT ITSELF, as a short claim the candidate
  should make, and must be specific to THIS question. A description of what the
  candidate should do ("explains ownership", "mentions the tradeoff") is useless
  for grading and will be rejected.
- Never begin an expected_point with: explains, describes, mentions, discusses,
  understands, identifies, distinguishes, demonstrates, covers, states, defines,
  gives, provides, lists, names.
- The points must differ from each other in substance. Two points that say the
  same thing about opposite cases ("a min-heap maintains the heap property", "a
  max-heap maintains the heap property") count as one point, not two.""",

    Mode.OPEN_ENDED: """\
Write ONE open-ended system design question.

Return JSON with exactly these keys:
  "text"              the prompt, one or two sentences
  "reference_answer"  what a strong answer covers, 3-5 sentences, spoken plainly
  "expected_points"   4-6 short strings; considerations a strong answer raises
  "follow_up_seeds"   2 probes that push on scale or failure modes

Rules:
- There is no single correct answer. expected_points are CONSIDERATIONS
  (tradeoffs, bottlenecks, failure modes), not facts.
- Answerable out loud in 3-4 minutes at a whiteboard level of detail.
- reference_answer is read aloud: no markdown, no bullets, no symbols.""",

    Mode.BEHAVIOURAL: """\
Write ONE behavioural interview question about the candidate's past experience.

Return JSON with exactly these keys:
  "text"              the question, phrased as "Tell me about a time..." or similar
  "reference_answer"  what a strong answer demonstrates, 2-4 sentences
  "expected_points"   4 strings covering: the situation, the candidate's specific
                      actions, the outcome, and what they learned or would change
  "follow_up_seeds"   2 probes that push for specifics if the answer is vague

Rules:
- There is NO correct answer. Never write anything that implies one exists.
- reference_answer describes the SHAPE of a strong response (concrete situation,
  clear personal ownership, measurable outcome, honest reflection) - it must not
  invent a story for the candidate.
- expected_points are STRUCTURAL, describing what the answer must contain:
  "names a specific situation with enough context to follow", "says what they
  personally did rather than what the team did", "gives a concrete outcome",
  "reflects on what they would change". These are the one case where describing
  the answer is correct, because there is no fact to check.""",
}

SYSTEM = (
    "You write interview questions for a spoken practice interview. "
    "You always reply with a single valid JSON object and nothing else."
)


def stem(text: str, n: int = 4) -> str:
    """The opening few words, used to detect stylistic monotony."""
    return " ".join(text.split()[:n]).lower().strip(",.?!")


def build_prompt(topic: T.Topic, subtopic: str, difficulty: int,
                 avoid_stems: list[str] | None = None) -> list[dict]:
    levels = {1: "warm-up, a junior candidate should get this",
              2: "straightforward, expected of a junior",
              3: "solid, expected of a mid-level engineer",
              4: "challenging, separates mid from senior",
              5: "hard, a senior-level stretch question"}
    user = (
        f"{SPEC[topic.mode]}\n\n"
        f"Topic: {topic.label}\n"
        f"Specifically about: {subtopic}\n"
        f"Difficulty: {difficulty} of 5 ({levels[difficulty]})\n\n"
        f"The question must be specifically about {subtopic!r}, not a generic "
        f"{topic.label} question."
    )
    # Left alone, the model opens four questions in six with "Can you explain
    # the ...". Each is fine on its own; a session of them sounds like a form.
    if avoid_stems and topic.mode is not Mode.BEHAVIOURAL:
        listed = ", ".join(f'"{x}"' for x in avoid_stems[:6])
        user += ("\n\nDo not begin the question with any of these openings, "
                 f"which are already overused: {listed}. Vary the form - ask "
                 "directly, pose a scenario, or ask for a comparison.")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# Validation — a 3B model will produce plausible-looking garbage some of the
# time, and a bad bank entry poisons every session that draws it.
# --------------------------------------------------------------------------- #

BANNED = re.compile(r"```|^\s*[-*]\s|\bwrite a function\b|\bimplement\b",
                    re.IGNORECASE | re.MULTILINE)

# An expected_point that merely narrates what the candidate should do
# ("explains paging") tells the Phase 3 grader nothing the question did not
# already say. Factual and design points have to assert real content.
VAGUE_POINT = re.compile(
    r"^\s*(explain|describ|mention|discuss|understand|identif|distinguish"
    r"|demonstrat|cover|state|show|talk|address|articulat|recogni|highlight"
    r"|defin|give|provid|list|name|outlin|note|clarif|illustrat|specif"
    r"|correctly|accurately|able to|knowledge of|awareness of)",
    re.IGNORECASE)


STOP = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or", "in",
    "on", "for", "that", "this", "it", "its", "with", "as", "by", "be", "you",
    "your", "they", "their", "them", "not", "but", "can", "will", "would",
    "when", "what", "how", "why", "which", "does", "do", "at", "from", "if",
    "one", "all", "more", "than", "into", "each", "used", "use", "using",
}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower())
            if w not in STOP and len(w) > 2}


def validate(d: dict, topic: T.Topic, subtopic: str = "") -> str | None:
    """Returns a rejection reason, or None if the entry is usable."""
    for key in ("text", "reference_answer", "expected_points", "follow_up_seeds"):
        if key not in d:
            return f"missing {key}"

    text = str(d["text"]).strip()
    ref = str(d["reference_answer"]).strip()
    pts = d["expected_points"]
    seeds = d["follow_up_seeds"]

    if not (15 <= len(text) <= 400):
        return f"question length {len(text)}"
    if not (40 <= len(ref) <= 1200):
        return f"reference_answer length {len(ref)}"
    if not isinstance(pts, list) or not (2 <= len(pts) <= 8):
        return "expected_points must be a list of 2-8 items"
    if any(not isinstance(p, str) or len(p) < 20 for p in pts):
        return "expected_points entries too short"

    # A point must be about THIS question. The model sometimes copies an example
    # out of the prompt wholesale — one trie question came back with a point
    # about mutex ownership — and a point that shares no content word with the
    # question or subtopic is the signature of that.
    context = content_words(f"{text} {topic.label} {subtopic}")
    for p in pts:
        if context and not (content_words(p) & context):
            return f"expected_point unrelated to the question: {p[:50]!r}"

    # Near-duplicate points inflate the denominator in Phase 3's coverage score,
    # so an answer that makes one real point gets credited for two.
    # Judge on what DIFFERS, not on overlap. The failure being caught is a point
    # duplicated with one word swapped ("a min-heap maintains..." / "a max-heap
    # maintains..."). An overlap ratio also flags legitimately parallel facts
    # like pre-order versus in-order traversal, which share most of their words
    # but say genuinely different things.
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            wa, wb = content_words(a), content_words(b)
            if wa and wb and len(wa ^ wb) <= 2:
                return f"near-duplicate expected_points: {a[:40]!r} / {b[:40]!r}"
    # Behavioural points legitimately describe answer structure; the other modes
    # must assert checkable content.
    if topic.mode is not Mode.BEHAVIOURAL:
        vague = [p for p in pts if VAGUE_POINT.match(p)]
        if len(vague) > len(pts) // 3:
            return f"{len(vague)}/{len(pts)} expected_points just restate the question"
    if not isinstance(seeds, list) or not seeds:
        return "follow_up_seeds missing"
    # Everything in reference_answer is read aloud by Piper.
    if BANNED.search(ref) or BANNED.search(text):
        return "contains markdown or a coding task"
    if topic.mode is Mode.BEHAVIOURAL and re.search(
            r"\bcorrect answer\b|\bthe right answer\b", ref, re.I):
        return "behavioural question implies a correct answer exists"
    return None


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def is_duplicate(text: str, existing: list[str]) -> bool:
    """Cheap token-overlap dedupe. The failure mode we care about is the model
    rewording the same question, which shares most of its content words."""
    words = set(normalise(text).split()) - {
        "what", "is", "the", "a", "an", "how", "why", "would", "you", "do",
        "does", "in", "of", "to", "and", "or", "for", "tell", "me", "about",
        "time", "when", "your", "that", "it", "with", "can", "explain",
    }
    if not words:
        return False
    for other in existing:
        o = set(normalise(other).split())
        if len(words & o) / len(words) > 0.65:
            return True
    return False


# --------------------------------------------------------------------------- #

# Failures whose only problem is the expected_points list. The question and the
# reference answer are usually fine, so resampling the whole thing throws away
# good work — repair just the points instead.
POINT_FAULTS = ("expected_points", "expected_point")

REPAIR_SYSTEM = (
    "You extract grading criteria from a model answer. You reply with a single "
    "JSON object and nothing else."
)


async def repair_points(llm: LLM, topic: T.Topic, d: dict, n: int = 4) -> dict:
    """Rewrite expected_points as content assertions, grounded in the answer.

    This is a much easier task than writing them from scratch, because every
    claim is already sitting in `reference_answer` — the model only has to split
    it up. In practice it converts most rejections into usable entries with one
    extra call, instead of a full resample that usually fails the same way.
    """
    user = (
        f"Question:\n{d['text']}\n\n"
        f"Model answer:\n{d['reference_answer']}\n\n"
        f"Break the model answer into exactly {n} separate claims. Each claim "
        f"must be one specific statement taken from the answer, written as the "
        f"statement itself, so it can be checked against what a candidate said.\n"
        f"Never begin a claim with: explains, describes, mentions, discusses, "
        f"understands, identifies, distinguishes, states, defines, gives, "
        f"lists, names, shows, covers.\n"
        f"Each claim must say something the others do not.\n"
        f'Reply with JSON: {{"points": ["...", "...", "...", "..."]}}'
    )
    raw = await llm.complete(
        [{"role": "system", "content": REPAIR_SYSTEM},
         {"role": "user", "content": user}],
        temperature=0.2, json_mode=True)
    points = json.loads(raw).get("points", [])
    if isinstance(points, list) and len(points) >= 2:
        return {**d, "expected_points": [str(p).strip() for p in points]}
    return d


async def generate_one(llm: LLM, topic: T.Topic, subtopic: str,
                       difficulty: int, attempts: int = 3,
                       avoid_stems: list[str] | None = None) -> Question | None:
    for attempt in range(attempts):
        try:
            raw = await llm.complete(
                build_prompt(topic, subtopic, difficulty, avoid_stems),
                # Offline, so we can afford variety; retries cost nothing here.
                temperature=0.8 if attempt else 0.6,
                json_mode=True,
            )
            d = json.loads(raw)
        except (json.JSONDecodeError, RuntimeError) as e:
            print(f"      attempt {attempt + 1}: bad response ({type(e).__name__})")
            continue

        reason = validate(d, topic, subtopic)
        if reason and any(f in reason for f in POINT_FAULTS):
            try:
                d = await repair_points(llm, topic, d)
                if repaired := validate(d, topic, subtopic):
                    print(f"      attempt {attempt + 1}: repair failed - {repaired}")
                    continue
                print(f"      attempt {attempt + 1}: points repaired")
                reason = None
            except (json.JSONDecodeError, RuntimeError, ValueError):
                print(f"      attempt {attempt + 1}: repair errored")
                continue

        if reason:
            print(f"      attempt {attempt + 1}: rejected - {reason}")
            continue

        return Question(
            topic=topic.key, difficulty=difficulty, mode=topic.mode,
            text=str(d["text"]).strip(),
            reference_answer=str(d["reference_answer"]).strip(),
            expected_points=[str(p).strip() for p in d["expected_points"]],
            follow_up_seeds=[str(s).strip() for s in d["follow_up_seeds"]][:3],
            subtopic=subtopic,
        )
    return None


async def generate_topic(llm: LLM, key: str, per_subtopic: int,
                         limit: int | None) -> None:
    topic = T.get(key)
    path = BANK / f"{key}.json"
    existing = load_questions(path)
    texts = [q.text for q in existing]

    subs = list(topic.subtopics)
    random.shuffle(subs)
    if limit:
        subs = subs[:limit]

    print(f"\n=== {topic.label} ({topic.mode}) ===")
    print(f"    {len(existing)} existing | {len(subs)} subtopics "
          f"x {per_subtopic} = {len(subs) * per_subtopic} target")

    added = rejected = dupes = 0
    t0 = time.perf_counter()

    spread = list(DIFFICULTIES)
    for i, sub in enumerate(subs, 1):
        offset = (i - 1) % len(spread)
        for n in range(per_subtopic):
            diff = DIFFICULTIES[n % len(DIFFICULTIES)] if per_subtopic > 1 \
                else random.choice(DIFFICULTIES)
            # Stems used twice or more already are fed back as things to avoid.
            overused = [st for st, c in Counter(
                stem(t) for t in texts).items() if c >= 2]
            q = await generate_one(llm, topic, sub, diff, avoid_stems=overused)
            if q is None:
                rejected += 1
                print(f"  [{i:2d}/{len(subs)}] d{diff} {sub[:44]:44s} FAILED")
                continue
            if is_duplicate(q.text, texts):
                dupes += 1
                print(f"  [{i:2d}/{len(subs)}] d{diff} {sub[:44]:44s} dupe")
                continue
            existing.append(q)
            texts.append(q.text)
            added += 1
            # Save after every accepted question. A full run takes hours, and
            # saving only at the end of a topic means an interruption throws
            # away everything generated since the last topic boundary.
            save_questions(path, existing)
            print(f"  [{i:2d}/{len(subs)}] d{diff} {sub[:44]:44s} {q.text[:60]}")

    save_questions(path, existing)
    print(f"    +{added} added, {dupes} dupes, {rejected} failed, "
          f"{len(existing)} total in {time.perf_counter() - t0:.0f}s -> {path.name}")


def review(key: str) -> None:
    """Print a topic's bank for human review. plan.md §2.2 asks for this."""
    qs = load_questions(BANK / f"{key}.json")
    topic = T.get(key)
    print(f"\n{topic.label} — {len(qs)} questions\n" + "=" * 72)
    for q in sorted(qs, key=lambda x: (x.difficulty, x.subtopic)):
        print(f"\n[d{q.difficulty}] {q.subtopic}\n  Q: {q.text}")
        print(f"  A: {q.reference_answer[:200]}")
        for p in q.expected_points:
            print(f"     - {p}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", action="append", default=[],
                    help="topic key; repeatable")
    ap.add_argument("--all", action="store_true", help="every topic")
    ap.add_argument("--per-subtopic", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap subtopics per topic (for a quick smoke run)")
    ap.add_argument("--review", action="store_true", help="print, do not generate")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    keys = list(T.TOPICS) if args.all else args.topic
    # Résumé questions are generated per-CV at session time, not banked.
    keys = [k for k in keys if not T.get(k).needs_resume]
    if not keys:
        ap.error("pass --topic KEY or --all")

    if args.review:
        for k in keys:
            review(k)
        return 0

    async with LLM() as llm:
        health = await llm.health()
        if not health["has_model"]:
            print("ollama does not have the configured model", file=sys.stderr)
            return 1
        for k in keys:
            await generate_topic(llm, k, args.per_subtopic, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
