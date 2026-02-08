"""certipatch.specs.balance_paren_14

Spec: BALANCE-PAREN-14 (fully enumerable, size 2^0 + ... + 2^14 = 32767)

Domain:
  All strings s over alphabet {'(', ')'} with length <= 14.

Question:
  "Is the parentheses string \"{s}\" balanced?"

Label:
  Yes iff stack-based parsing never underflows and ends at depth 0.

Enumeration order (canonical):
  length L from 0..14:
    enumerate all bitstrings of length L where 0='(' and 1=')' in increasing integer order.

Hashing:
  SHA256 of newline-joined canonical prompts (UTF-8, LF endings).

This spec is intended as an algorithmic stress test. The project MUST report
fail-closed results if 0 failures cannot be achieved under collateral constraints.

This file is a scaffold. Implementations are omitted.
"""

from __future__ import annotations

from typing import Iterator, Mapping

from . import SpecExample, boolqa_prompt

SPEC_ID = "balance_paren_14"


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

    max_len = int(spec_cfg.get("max_len", 14))
    if not (0 <= max_len <= 14):
        raise ValueError(f"Invalid max_len for {SPEC_ID}: {max_len} expected within [0, 14].")

    for L in range(0, max_len + 1):
        for b in range(0, 2**L):
            # Big-endian bit order yields lexicographic order with '(' < ')'.
            bits = [((b >> (L - 1 - j)) & 1) for j in range(L)]
            s = "".join("(" if bit == 0 else ")" for bit in bits)
            q = question(s)
            prompt = boolqa_prompt(wrapper_line, q, suffix)
            yield SpecExample(
                prompt=prompt,
                label=label(s),
                meta={
                    "id": f"balance14/L={L}/b={b}",
                    "L": str(L),
                    "b": str(b),
                },
            )


def is_balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
        else:
            return False
    return depth == 0


def label(s: str) -> int:
    return 1 if is_balanced(s) else 0


def question(s: str) -> str:
    return f'Is the parentheses string "{s}" balanced?'
