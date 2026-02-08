"""certipatch.specs.parity_4d

Spec: PARITY-4D (fully enumerable, size 10,000)

Domain:
  n in [0..9999], represented canonically as n_str = str(n) (no zero padding).

Question:
  "Is {n} even?"

Label:
  Yes iff n % 2 == 0

Enumeration order (canonical):
  for n in 0..9999:
    emit

Hashing:
  SHA256 of newline-joined canonical prompts (UTF-8, LF endings).

This file is a scaffold. Implementations are omitted.
"""

from __future__ import annotations

from typing import Iterator, Mapping

from . import SpecExample, boolqa_prompt

SPEC_ID = "parity_4d"


def iter_domain(cfg: Mapping) -> Iterator[SpecExample]:
    """Yield all examples in canonical order."""
    if "gate" not in cfg or not isinstance(cfg["gate"], Mapping):
        raise ValueError("cfg['gate'] must be provided for spec prompt construction.")
    wrapper_line = cfg["gate"].get("wrapper_line")
    suffix = cfg["gate"].get("suffix")
    if not isinstance(wrapper_line, str) or not wrapper_line:
        raise ValueError("cfg['gate']['wrapper_line'] must be a non-empty string.")
    if not isinstance(suffix, str) or not suffix:
        raise ValueError("cfg['gate']['suffix'] must be a non-empty string.")

    if "specs" not in cfg or not isinstance(cfg["specs"], Mapping):
        raise ValueError("cfg['specs'] must be provided.")
    spec_cfg = cfg["specs"].get(SPEC_ID)
    if not isinstance(spec_cfg, Mapping):
        raise ValueError(f"cfg['specs']['{SPEC_ID}'] must be provided.")

    n_min = int(spec_cfg.get("n_min", 0))
    n_max = int(spec_cfg.get("n_max", 9999))
    if not (0 <= n_min <= n_max <= 9999):
        raise ValueError(
            f"Invalid n range for {SPEC_ID}: [{n_min}, {n_max}] expected within [0, 9999]."
        )

    for n in range(n_min, n_max + 1):
        q = question(n)
        prompt = boolqa_prompt(wrapper_line, q, suffix)
        yield SpecExample(
            prompt=prompt,
            label=label(n),
            meta={
                "id": f"parity4d/n={n}",
                "n": str(n),
            },
        )


def label(n: int) -> int:
    return 1 if (n % 2 == 0) else 0


def question(n: int) -> str:
    nn = str(n)
    return f"Is {nn} even?"
