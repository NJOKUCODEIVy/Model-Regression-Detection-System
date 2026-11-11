"""LLM-as-judge: rates how well each generated summary captures the customer's email, from 1 to 5.

The judge uses a different model from the classifier (JUDGE_MODEL, default openai/gpt-oss-20b).
That keeps a model from grading its own work, and on Groq each model has its own rate limit,
so judge calls don't eat into the classifier's budget.
"""

import os

from pydantic import BaseModel, Field

from src.llm import LLM, extract_json

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")

JUDGE_SYSTEM_PROMPT = """You grade one-sentence summaries of customer support emails.
You get the customer's email, a reference summary written by a person, and a candidate summary.
The reference is one good answer, not the only one: different wording is fine.

Rate how well the candidate captures what the customer needs:
5 = the customer's main request and the key details, accurate, in one sentence
4 = accurate, but misses a minor detail
3 = partly right: misses an important detail, or adds a small inaccuracy
2 = mostly off: vague, or misreads the request
1 = wrong, unrelated, or empty

Respond with ONLY a JSON object and no other text:
{"score": <1-5>, "reason": "<one short sentence>"}"""


class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str


async def judge_summary(llm: LLM, email: str, reference: str, candidate: str) -> dict:
    """Returns the judge's score and reason, plus the call's latency and token counts."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Email:\n{email}\n\nReference summary:\n{reference}\n\nCandidate summary:\n{candidate}"},
    ]
    call = await llm.chat(JUDGE_MODEL, messages)
    result = JudgeResult.model_validate(extract_json(call.text))
    return {
        "judge_score": result.score,
        "judge_reason": result.reason,
        "judge_latency_ms": call.latency_ms,
        "judge_prompt_tokens": call.prompt_tokens,
        "judge_completion_tokens": call.completion_tokens,
    }