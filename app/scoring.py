from collections import defaultdict
from typing import Dict, List, Tuple

from .models import Answer
from .questions import QUESTIONS

_Q_TO_COMP = {q["id"]: q["competency"] for q in QUESTIONS}
GAP_THRESHOLD = 4.0


def compute_scores(answers: List[Answer]) -> Tuple[Dict[str, float], float]:
    buckets = defaultdict(list)
    for a in answers:
        buckets[_Q_TO_COMP[a.question_id]].append(a.value)
    scores = {c: round(sum(v) / len(v), 2) for c, v in buckets.items()}
    overall = round(sum(scores.values()) / len(scores), 2)
    return scores, overall


def level_for(overall: float) -> str:
    if overall < 2.0:
        return "beginner"
    if overall < 3.5:
        return "developing"
    if overall < 4.5:
        return "proficient"
    return "advanced"


def split(scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
    strengths = sorted([c for c, s in scores.items() if s >= GAP_THRESHOLD])
    gaps = sorted([c for c, s in scores.items() if s < GAP_THRESHOLD], key=lambda c: scores[c])
    return strengths, gaps 