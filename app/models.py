from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class Answer(BaseModel):
    question_id: int = Field(ge=1, le=10)
    value: int = Field(ge=1, le=5)


class SubmitRequest(BaseModel):
    user_name: str = Field(min_length=1, max_length=80)
    answers: List[Answer] = Field(min_length=10, max_length=10)

    @field_validator("user_name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("name is empty")
        return v

    @field_validator("answers")
    @classmethod
    def unique_questions(cls, v: List[Answer]) -> List[Answer]:
        ids = sorted(a.question_id for a in v)
        if ids != list(range(1, 11)):
            raise ValueError("answers must cover question ids 1..10 exactly once")
        return v


class Gap(BaseModel):
    competency: str
    label: str
    score: float
    why: str
    recommendation: str


class GapReport(BaseModel):
    overall_score: float
    overall_level: Literal["beginner", "developing", "proficient", "advanced"]
    strengths: List[str]
    gaps: List[Gap]
    summary: str
    generated_by: Literal["llm", "fallback"]


# What we ask the LLM to return (narrow on purpose)
class LLMGap(BaseModel):
    competency: str
    why: str
    recommendation: str


class LLMOutput(BaseModel):
    summary: str
    gaps: List[LLMGap]