"""certipatch.specs.compare_2d

Spec: COMPARE-2D (fully enumerable, size 10,000)

Domain:
  a, b in [0..99] with zero-padded width=2

Question:
  "Is {a} greater than {b}?"

Label:
  Yes iff a > b

Enumeration order (canonical):
  for a in 0..99:
    for b in 0..99:
      emit

Hashing:
  The domain hash is SHA256 of the newline-joined canonical prompts (UTF-8, LF endings).

This file is a scaffold. Implementations are omitted.
"""

from __future__ import annotations

from typing import Iterator, Mapping

from . import SpecExample, boolqa_prompt

SPEC_ID = "compare_2d"


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

    a_min = int(spec_cfg.get("a_min", 0))
    a_max = int(spec_cfg.get("a_max", 99))
    b_min = int(spec_cfg.get("b_min", 0))
    b_max = int(spec_cfg.get("b_max", 99))

    if not (0 <= a_min <= a_max <= 99):
        raise ValueError(
            f"Invalid a range for {SPEC_ID}: [{a_min}, {a_max}] expected within [0, 99]."
        )
    if not (0 <= b_min <= b_max <= 99):
        raise ValueError(
            f"Invalid b range for {SPEC_ID}: [{b_min}, {b_max}] expected within [0, 99]."
        )

    for a in range(a_min, a_max + 1):
        for b in range(b_min, b_max + 1):
            aa = f"{a:02d}"
            bb = f"{b:02d}"
            q = question(a, b)
            prompt = boolqa_prompt(wrapper_line, q, suffix)
            yield SpecExample(
                prompt=prompt,
                label=label(a, b),
                meta={
                    "id": f"compare2d/a={aa}/b={bb}",
                    "a": aa,
                    "b": bb,
                },
            )


def label(a: int, b: int) -> int:
    """Return 1 if a>b else 0."""
    return 1 if a > b else 0


def question(a: int, b: int) -> str:
    """Return the question string."""
    aa = f"{a:02d}"
    bb = f"{b:02d}"
    return f"Is {aa} greater than {bb}?"
