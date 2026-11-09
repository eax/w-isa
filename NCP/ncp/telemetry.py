"""Token-level telemetry and explicit task scoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import re


@dataclass(frozen=True)
class TaskCase:
    case_id: str
    prompt: str
    expected_answer: str
    answer_pattern: str = r"(?im)^\s*Final answer:\s*([^\n]+)"

    def __post_init__(self):
        if not self.case_id or not self.prompt or not isinstance(self.expected_answer, str):
            raise ValueError("task ID, prompt and string expected answer are required")
        if re.compile(self.answer_pattern).groups < 1:
            raise ValueError("answer_pattern must capture the answer in group 1")

    def score(self, text: str) -> bool:
        matches = list(re.finditer(self.answer_pattern, text))
        answer = matches[-1].group(1) if matches else text
        return answer.strip().rstrip(".").strip().casefold() == self.expected_answer.strip().casefold()


def repetition_metrics(tokens: list[int], ngram_size: int = 4) -> dict:
    if ngram_size < 1:
        raise ValueError("ngram size must be positive")
    grams = [tuple(tokens[i:i + ngram_size]) for i in range(max(0, len(tokens) - ngram_size + 1))]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values())
    first_repeat = None
    max_run = 1 if grams else 0
    runs = [1] * len(grams)
    for i in range(ngram_size, len(grams)):
        if grams[i] == grams[i - ngram_size]:
            runs[i] = runs[i - ngram_size] + 1
            max_run = max(max_run, runs[i])
            if first_repeat is None:
                first_repeat = i
    return {"ngram_size": ngram_size, "ngram_count": len(grams), "repeated_ngrams": repeated,
            "repeat_fraction": repeated / len(grams) if grams else 0.0,
            "max_consecutive_repeats": max_run, "first_adjacent_repeat_token": first_repeat}


@dataclass
class Observation:
    case_id: str
    text: str
    token_ids: list[int]
    correct: bool
    entropy_nats: list[float]
    repetition: dict
    token_budget_reached: bool
    seconds: float
    matched_rule: str | None = None
    match_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def summarize(observations: list[Observation]) -> dict:
    if not observations:
        raise ValueError("evaluation requires at least one case")
    return {"cases": len(observations), "correct": sum(o.correct for o in observations),
            "accuracy": sum(o.correct for o in observations) / len(observations),
            "mean_repeat_fraction": sum(o.repetition["repeat_fraction"] for o in observations) / len(observations),
            "budget_hits": sum(o.token_budget_reached for o in observations),
            "generation_seconds": sum(o.seconds for o in observations)}