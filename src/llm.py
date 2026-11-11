"""Shared async LLM calls for the eval: one client, a concurrency limit, per-model rate limiting,
retries, and the latency and token counts of every call.

Settings (environment variables):
    EVAL_CONCURRENCY      requests in flight at once (default 4)
    REQUESTS_PER_MINUTE   requests started per minute, per model (default 25; Groq's free tier allows about 30)
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI

LLM_BASE_URL = "https://api.groq.com/openai/v1"
MAX_ATTEMPTS = 6
RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)


@dataclass
class CallResult:
    text: str
    latency_ms: int                  # the successful attempt only, not time spent waiting on rate limits
    prompt_tokens: int | None
    completion_tokens: int | None
    attempts: int


class RateLimiter:
    """Spaces out request starts so one model gets at most `per_minute` requests a minute."""

    def __init__(self, per_minute: float):
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_start - now)
            self.next_start = max(now, self.next_start) + self.interval
        if delay:
            await asyncio.sleep(delay)


def retry_delay(err: Exception, attempt: int) -> float:
    """Use the server's Retry-After header when it sends one, otherwise back off exponentially."""
    response = getattr(err, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    try:
        return min(60.0, float(header))
    except (TypeError, ValueError):
        return min(30.0, 2.0 ** attempt)


class LLM:
    """Create inside a running event loop (in an async function) and close when done."""

    def __init__(self, concurrency: int | None = None, per_minute: float | None = None):
        self.client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=os.environ["GROQ_API_KEY"], max_retries=0)
        self.semaphore = asyncio.Semaphore(concurrency or int(os.getenv("EVAL_CONCURRENCY", "4")))
        self.per_minute = per_minute or float(os.getenv("REQUESTS_PER_MINUTE", "25"))
        self.limiters: dict[str, RateLimiter] = {}  # rate limits are per model, so each model gets its own

    async def chat(self, model: str, messages: list[dict], temperature: float = 0.0) -> CallResult:
        limiter = self.limiters.setdefault(model, RateLimiter(self.per_minute))
        async with self.semaphore:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                await limiter.wait()
                start = time.perf_counter()
                try:
                    response = await self.client.chat.completions.create(
                        model=model, messages=messages, temperature=temperature)
                except RETRYABLE as err:
                    if attempt == MAX_ATTEMPTS:
                        raise
                    await asyncio.sleep(retry_delay(err, attempt))
                    continue
                usage = response.usage
                return CallResult(
                    text=response.choices[0].message.content or "",
                    latency_ms=round((time.perf_counter() - start) * 1000),
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    attempts=attempt,
                )
        raise RuntimeError("unreachable")

    async def close(self) -> None:
        await self.client.close()


def extract_json(text: str) -> dict:
    """Pull the JSON object out of a model reply, ignoring any text around it."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model response: {text!r}")
    return json.loads(text[start : end + 1])