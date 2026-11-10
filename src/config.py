"""The interface contract for the classifier: what goes in and what comes out."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Category = Literal["billing", "technical", "account", "general"]


class ClassificationResult(BaseModel):
    """Structured output the classifier must return."""

    category: Category
    summary: str = Field(min_length=1)


class FewShotExample(BaseModel):
    email: str
    output: ClassificationResult  # validated too, so a typo in the YAML fails loudly


class PromptConfig(BaseModel):
    """One versioned prompt, loaded from a YAML file in /prompts."""

    version: str
    created_at: datetime
    model: str
    temperature: float = 0.0
    system_prompt: str
    few_shot_examples: list[FewShotExample] = []

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptConfig":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
        