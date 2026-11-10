"""The LLM feature under test: a customer support email classifier."""

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.config import ClassificationResult, PromptConfig

load_dotenv()

LLM_BASE_URL = "https://api.groq.com/openai/v1"


def get_client() -> OpenAI:
    # max_retries: wait and retry automatically if the free tier's rate limit is hit
    return OpenAI(base_url=LLM_BASE_URL, api_key=os.environ["GROQ_API_KEY"], max_retries=6)


def build_messages(email: str, config: PromptConfig) -> list[dict]:
    """System prompt, then each few-shot example as a user/assistant pair, then the real email."""
    messages = [{"role": "system", "content": config.system_prompt}]
    for example in config.few_shot_examples:
        messages.append({"role": "user", "content": example.email})
        messages.append({"role": "assistant", "content": example.output.model_dump_json()})
    messages.append({"role": "user", "content": email})
    return messages


def parse_response(text: str) -> ClassificationResult:
    """Pull the JSON object out of the reply and validate it against the contract."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model response: {text!r}")
    return ClassificationResult.model_validate_json(text[start : end + 1])


def classify_email(email: str, config: PromptConfig) -> ClassificationResult:
    response = get_client().chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        messages=build_messages(email, config),
    )
    return parse_response(response.choices[0].message.content or "")


if __name__ == "__main__":
    config = PromptConfig.from_yaml("prompts/v1.yaml")
    sample = "Hi, I updated my credit card last week but my new invoice still shows the old card."
    result = classify_email(sample, config)
    print(f"[{config.version}] {result.category}: {result.summary}")