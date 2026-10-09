from app.models import Answer
from app.scoring import compute_scores, level_for, priority_for, split


def _answers(vals):
    return [Answer(question_id=i + 1, value=v) for i, v in enumerate(vals)]


def test_scores_and_overall():
    scores, overall = compute_scores(_answers([5, 4, 2, 3, 4, 4, 2, 3, 5, 5]))
    assert scores["active_listening"] == 4.5
    assert scores["clarity"] == 2.5
    assert overall == 3.7


def test_levels():
    assert level_for(1.9) == "beginner"
    assert level_for(2.0) == "developing"
    assert level_for(3.5) == "proficient"
    assert level_for(4.5) == "advanced"


def test_priority_and_split():
    assert priority_for(2.0) == "high"
    assert priority_for(3.0) == "medium"
    assert priority_for(3.8) == "low"
    scores, _ = compute_scores(_answers([5, 4, 2, 3, 4, 4, 2, 3, 5, 5]))
    strengths, gaps = split(scores)
    assert set(strengths) == {"active_listening", "feedback", "written_communication"}
    assert set(gaps) == {"clarity", "conflict_handling"}