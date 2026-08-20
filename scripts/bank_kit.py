r"""Shared machinery for the hand-written question banks.

Every rule here was learned from the generated banks failing it. They are
enforced at build time rather than written in a comment, because "keep the
questions short" is a promise that decays the moment someone adds "just one
more" entry.

* **One question, one sentence, under 20 words.** If it needs an "and", it is
  two questions; the second becomes a follow-up seed.
* **Three to five expected points, each under 12 words.** The grader asks "did
  they state this?" once per point, so a point that is really three facts in a
  trench coat grades as a coin flip.
* **Reference answers are spoken prose**, because they are read aloud when the
  candidate is wrong. No code blocks, no symbols, and no LaTeX.
* **Openings vary.** The generated Python bank opened five of eight questions
  with "Can you explain the", which makes an interview sound like a form being
  read out.

Used by `build_python_bank.py` and `build_applied_science_banks.py`.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from coach.config import BANK                              # noqa: E402
from coach.schema import (Mode, Question, load_questions,  # noqa: E402
                          save_questions)

# (difficulty, subtopic, question, reference, expected points, follow-up seeds)
Entry = tuple[int, str, str, str, list[str], list[str]]

MAX_QUESTION_WORDS = 20
MAX_POINT_WORDS = 12
MIN_POINTS, MAX_POINTS = 3, 5

# Things that read fine on a page and badly out loud. The reference answer is
# spoken by Piper when the candidate gets it wrong, and it pronounces none of
# these usefully.
_UNSPEAKABLE = re.compile(r"[`_*#\\{}<>|]|\$\$|\bO\(|\[\d")

DIFFICULTY_LABEL = {1: "basic", 2: "easy", 3: "medium", 4: "hard", 5: "stretch"}


class BankError(SystemExit):
    pass


def build(topic: str, mode: Mode, entries: list[Entry]) -> list[Question]:
    """Validate and convert one topic's entries. Raises on any rule violation."""
    seen: set[str] = set()
    out: list[Question] = []

    for diff, sub, text, ref, points, seeds in entries:
        where = f"[{topic} d{diff} {sub!r}]"

        key = re.sub(r"[^a-z ]", "", text.lower()).strip()
        if key in seen:
            raise BankError(f"{where} duplicate question: {text}")
        seen.add(key)

        words = len(text.split())
        if words > MAX_QUESTION_WORDS:
            raise BankError(f"{where} question is {words} words: {text}")
        if text.count("?") > 1:
            raise BankError(f"{where} two questions in one: {text}")
        if not text.rstrip().endswith(("?", ".")):
            raise BankError(f"{where} question has no terminator: {text}")

        if not MIN_POINTS <= len(points) <= MAX_POINTS:
            raise BankError(
                f"{where} has {len(points)} expected points, "
                f"want {MIN_POINTS}-{MAX_POINTS}: {text}")
        for p in points:
            if len(p.split()) > MAX_POINT_WORDS:
                raise BankError(f"{where} expected point too long: {p}")
        if len(set(points)) != len(points):
            raise BankError(f"{where} repeated expected point: {text}")

        if len(ref.split()) < 20:
            raise BankError(f"{where} reference answer is too thin: {text}")
        if bad := _UNSPEAKABLE.search(ref):
            raise BankError(
                f"{where} reference answer contains {bad.group()!r}, which is "
                f"read aloud badly: {text}")

        out.append(Question(
            topic=topic, difficulty=diff, mode=mode, text=text,
            reference_answer=ref, expected_points=points,
            follow_up_seeds=seeds, subtopic=sub,
        ))

    _check_variety(topic, out)
    return out


def _check_variety(topic: str, qs: list[Question]) -> None:
    """No single opening may account for more than a third of a topic."""
    if len(qs) < 6:
        return
    stems = Counter(" ".join(q.text.split()[:3]).lower() for q in qs)
    stem, n = stems.most_common(1)[0]
    if n > len(qs) / 3:
        raise BankError(
            f"[{topic}] {n} of {len(qs)} questions open with {stem!r} — vary "
            f"the phrasing, or the interview sounds like a form being read out")


def check_no_cross_topic_duplicates(built: dict[str, list[Question]]) -> None:
    """No two topics may ask the same question.

    `build` dedupes within a topic, which was enough while the topics were
    disjoint. Once Retrieval was split out of Generative AI, and Model
    Evaluation out of Machine Learning, the boundaries stopped being obvious —
    "how do you evaluate a generative system" is a fair question for either of
    two banks, and if both hold it a mixed interview asks it twice.

    Near-misses matter as much as exact ones, so this compares the content words
    rather than the strings.
    """
    # Banks already on disk count too. Python is built by its own script, so
    # comparing only what this run produced would miss a clash with it — and
    # whichever builder runs second is exactly where such a clash appears.
    on_disk: dict[str, list[Question]] = {}
    for path in sorted(BANK.glob("*.json")):
        if path.stem not in built:
            on_disk[path.stem] = load_questions(path)

    seen: dict[frozenset[str], tuple[str, str]] = {}
    clashes: list[str] = []

    for topic, questions in {**on_disk, **built}.items():
        for q in questions:
            key = frozenset(_content_words(q.text))
            if len(key) < 3:
                continue
            if (prev := seen.get(key)) and prev[0] != topic:
                clashes.append(
                    f"  {prev[0]}: {prev[1]}\n  {topic}: {q.text}")
            else:
                seen[key] = (topic, q.text)

    if clashes:
        raise BankError(
            "the same question appears in more than one topic:\n"
            + "\n\n".join(clashes))


_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "you",
    "your", "and", "or", "of", "to", "in", "on", "for", "that", "this", "it",
    "its", "with", "as", "by", "be", "what", "why", "how", "when", "would",
    "which", "can", "not", "but", "at", "from", "if", "so", "there", "them",
    "they", "we", "i", "s", "t", "about", "between", "difference", "whats",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def write(topic: str, questions: list[Question]) -> Path:
    path = BANK / f"{topic}.json"
    save_questions(path, questions)
    return path


def report(topic: str, questions: list[Question]) -> None:
    spread = Counter(q.difficulty for q in questions)
    easy = spread[1] + spread[2]
    lengths = [len(q.text.split()) for q in questions]
    print(f"\n{topic}  —  {len(questions)} questions, "
          f"{len({q.subtopic for q in questions})} subtopics")
    for d in sorted(spread):
        print(f"    {d} {DIFFICULTY_LABEL[d]:<8} {spread[d]:>3}  {'#' * spread[d]}")
    print(f"    basic+easy {easy}/{len(questions)} "
          f"({easy * 100 // len(questions)}%), "
          f"longest question {max(lengths)} words")


def summarise(built: dict[str, list[Question]]) -> None:
    total = sum(len(v) for v in built.values())
    spread: Counter[int] = Counter()
    for qs in built.values():
        spread.update(q.difficulty for q in qs)
    print(f"\n{'=' * 60}\n{total} questions written across {len(built)} topics")
    for d in sorted(spread):
        print(f"  {d} {DIFFICULTY_LABEL[d]:<8} {spread[d]:>4}")
    easy = spread[1] + spread[2]
    print(f"  basic or easy: {easy}/{total} ({easy * 100 // total}%)")
