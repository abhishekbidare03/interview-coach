"""The interview domain model.

The central decision here is `Mode` (plan.md §2.1). A single "is this answer
correct?" grader is wrong: "what is a deadlock" has a correct answer, "tell me
about a conflict with a teammate" does not, and grading the second as if it were
the first produces feedback that is worse than none. So every question declares
how it must be judged, and the evaluator dispatches on that field.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path


class Mode(StrEnum):
    """How an answer to this question must be judged."""

    FACTUAL = "factual"
    """Has a defensible correct answer. Verdict: correct / partial / incorrect.
    When wrong, the coach states the correct answer — the user's R5/R6 flow."""

    BEHAVIOURAL = "behavioural"
    """No correct answer exists. Judged on STAR structure, specificity, and
    ownership. Feedback names the missing component, never a 'right answer'."""

    OPEN_ENDED = "open_ended"
    """Design and judgement questions. Judged against a checklist of
    considerations a strong answer would raise (tradeoffs, scale, failure)."""


class Verdict(StrEnum):
    # factual
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    # behavioural / open-ended
    STRONG = "strong"
    ADEQUATE = "adequate"
    WEAK = "weak"
    # transcript was not trustworthy enough to judge (plan.md §2.4.2)
    UNCLEAR = "unclear"

    @property
    def is_good(self) -> bool:
        return self in (Verdict.CORRECT, Verdict.STRONG)

    @property
    def is_poor(self) -> bool:
        return self in (Verdict.INCORRECT, Verdict.WEAK)


VERDICTS_FOR: dict[Mode, tuple[Verdict, ...]] = {
    Mode.FACTUAL: (Verdict.CORRECT, Verdict.PARTIAL, Verdict.INCORRECT),
    Mode.BEHAVIOURAL: (Verdict.STRONG, Verdict.ADEQUATE, Verdict.WEAK),
    Mode.OPEN_ENDED: (Verdict.STRONG, Verdict.ADEQUATE, Verdict.WEAK),
}


def slug(text: str, n: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:n]
    return "-".join(words) or uuid.uuid4().hex[:8]


@dataclass
class Question:
    """One bank entry. Generated offline, reviewed by hand, adapted at runtime."""

    topic: str
    difficulty: int            # 1 (warm-up) .. 5 (senior / stretch)
    mode: Mode
    text: str

    reference_answer: str = ""
    """What a strong answer contains. For FACTUAL this is *the* answer and is
    read aloud when the candidate is wrong. For the other modes it is an
    exemplar, never something to recite at the candidate."""

    expected_points: list[str] = field(default_factory=list)
    """The specific things a complete answer covers. This is what makes grading
    tractable for a 3B model — it checks coverage against a list rather than
    reasoning about correctness from scratch."""

    follow_up_seeds: list[str] = field(default_factory=list)
    """Probes to deepen the question when the answer was strong or evasive."""

    subtopic: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        self.mode = Mode(self.mode)
        self.difficulty = max(1, min(5, int(self.difficulty)))
        if not self.id:
            self.id = f"{self.topic}-{self.difficulty}-{slug(self.text)}"

    @property
    def verdicts(self) -> tuple[Verdict, ...]:
        return VERDICTS_FOR[self.mode]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = str(self.mode)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PlannedQuestion:
    """A question's slot in a session, before it is asked."""
    question: Question
    position: int
    planned_difficulty: int


@dataclass
class Blueprint:
    """The plan for one interview, built before the first question is asked.

    plan.md §2.3 puts this work off the latency path deliberately: the candidate
    sees a "preparing your interview" state, so it can afford to be slow and
    thoughtful in a way that mid-conversation reasoning cannot.
    """

    title: str
    topics: list[str]
    questions: list[PlannedQuestion]
    difficulty_curve: list[int]
    vocabulary: str = ""       # fed to Whisper as initial_prompt (§2.4.1)
    opening_line: str = ""
    mixed: bool = False

    @property
    def length(self) -> int:
        return len(self.questions)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "topics": self.topics,
            "difficulty_curve": self.difficulty_curve,
            "vocabulary": self.vocabulary,
            "opening_line": self.opening_line,
            "mixed": self.mixed,
            "questions": [
                {"position": p.position, "planned_difficulty": p.planned_difficulty,
                 **p.question.to_dict()}
                for p in self.questions
            ],
        }


# --------------------------------------------------------------------------- #
# Bank persistence
# --------------------------------------------------------------------------- #

def load_questions(path: Path) -> list[Question]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Question.from_dict(d) for d in data]


def save_questions(path: Path, questions: list[Question]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([q.to_dict() for q in questions], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
