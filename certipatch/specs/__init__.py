"""certipatch.specs

Spec generators are deterministic offline domains with programmatic labels.

All specs MUST emit prompts using the shared BoolQA wrapper:
  Instruction: Answer with a single token: Yes or No.
  Question: ...
  Answer:

The wrapper line and suffix MUST be configurable and shared across all specs.

This package provides:
- A minimal dataclass `SpecExample`
- Registration helpers
- Deterministic canonical enumeration rules

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal


@dataclass(frozen=True)
class SpecExample:
    """A single labeled prompt instance."""

    prompt: str
    label: int  # 1 for Yes, 0 for No
    meta: Dict[str, str]  # deterministic metadata (e.g., a,b values, stratum)


SpecId = Literal["compare_2d", "parity_4d", "balance_paren_14", "compare_6d_strat"]


def boolqa_prompt(wrapper_line: str, question: str, suffix: str) -> str:
    """Construct the canonical BoolQA wrapper prompt."""
    return f"{wrapper_line}\nQuestion: {question}\n{suffix}"
