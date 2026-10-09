import asyncio
import re
import time

from app.llm import generate_report
from app.models import Answer
from app.scoring import compute_scores, split

ARABIC = re.compile(r"[\u0600-\u06FF]")
PROFILES = {
    "weak": [1] * 10,
    "mixed": [5, 4, 2, 3, 4, 4, 2, 3, 5, 5],
    "borderline": [4, 4, 3, 3, 4, 4, 4, 4, 3, 4],
    "strong": [5] * 10,   # no gaps -> no LLM call by design
}


async def run():
    failures = 0
    for name, vals in PROFILES.items():
        answers = [Answer(question_id=i + 1, value=v) for i, v in enumerate(vals)]
        scores, overall = compute_scores(answers)
        _, gaps = split(scores)
        t0 = time.time()
        rep = await generate_report(scores, overall)
        dt = time.time() - t0

        expected = "fallback" if not gaps else "llm"
        checks = {
            "source": rep.generated_by == expected,
            "scores_from_code": all(g.score == scores[g.competency] for g in rep.gaps),
            "gaps_match": sorted(g.competency for g in rep.gaps) == sorted(gaps),
            "arabic": all(ARABIC.search(t) for t in [rep.summary] + [g.why for g in rep.gaps]),
        }
        ok = all(checks.values())
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name:10} overall={overall} by={rep.generated_by} {dt:.1f}s {checks if not ok else ''}")
        if ok and gaps:
            print("   summary:", rep.summary)
            print("   rec:", rep.gaps[0].recommendation)
    print("\nALL OK" if failures == 0 else f"\n{failures} FAILED")


asyncio.run(run())