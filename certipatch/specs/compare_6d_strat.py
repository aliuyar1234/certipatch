"""certipatch.specs.compare_6d_strat

Spec: COMPARE-6D-STRAT (non-enumerable; coverage-bounded certificate)

Domain:
  a, b are 6-digit strings in [000000..999999]; full size is 10^12 and is not enumerable.

Certified scope:
  A fixed, deterministic coverage plan with strata:
    - S_k: most-significant differing digit index k in {0..5}
    - S_eq: a == b
    - S_near: |a-b| in a fixed delta set (default: {1,2,5,10})
    - S_ext: extremes including {000000,000001,999998,999999} paired with deterministic samples

Coverage plan parameters are part of config and MUST be hashed into the certificate.

This file is a scaffold. Implementations are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from . import SpecExample, boolqa_prompt

SPEC_ID = "compare_6d_strat"


@dataclass(frozen=True)
class CoveragePlan:
    """Deterministic coverage plan parameters."""

    per_msd_stratum_n: int
    eq_n: int
    near_n: int
    ext_n: int
    seed: int
    near_deltas: List[int]
    additional_random_n: int
    local_perturb_k: int


def _require_gate(cfg: Mapping) -> Tuple[str, str]:
    if "gate" not in cfg or not isinstance(cfg["gate"], Mapping):
        raise ValueError("cfg['gate'] must be provided for spec prompt construction.")
    wrapper_line = cfg["gate"].get("wrapper_line")
    suffix = cfg["gate"].get("suffix")
    if not isinstance(wrapper_line, str) or not wrapper_line:
        raise ValueError("cfg['gate']['wrapper_line'] must be a non-empty string.")
    if not isinstance(suffix, str) or not suffix:
        raise ValueError("cfg['gate']['suffix'] must be a non-empty string.")
    return wrapper_line, suffix


def _require_plan(cfg: Mapping) -> CoveragePlan:
    if "specs" not in cfg or not isinstance(cfg["specs"], Mapping):
        raise ValueError("cfg['specs'] must be provided.")
    spec_cfg = cfg["specs"].get(SPEC_ID)
    if not isinstance(spec_cfg, Mapping):
        raise ValueError(f"cfg['specs']['{SPEC_ID}'] must be provided.")
    coverage_cfg = spec_cfg.get("coverage")
    if not isinstance(coverage_cfg, Mapping):
        raise ValueError(f"cfg['specs']['{SPEC_ID}']['coverage'] must be provided.")

    near_deltas = list(coverage_cfg.get("near_deltas", [1, 2, 5, 10]))
    if not near_deltas:
        raise ValueError("near_deltas must be non-empty.")
    if any(int(d) <= 0 for d in near_deltas):
        raise ValueError("near_deltas must be positive integers.")

    return CoveragePlan(
        per_msd_stratum_n=int(coverage_cfg["per_msd_stratum_n"]),
        eq_n=int(coverage_cfg["eq_n"]),
        near_n=int(coverage_cfg["near_n"]),
        ext_n=int(coverage_cfg["ext_n"]),
        seed=int(coverage_cfg["seed"]),
        near_deltas=[int(d) for d in near_deltas],
        additional_random_n=int(coverage_cfg.get("additional_random_n", 0)),
        local_perturb_k=int(coverage_cfg.get("local_perturb_k", 0)),
    )


def _digits_base10(n: int, width: int) -> List[int]:
    if width < 0:
        raise ValueError("width must be >= 0")
    if width == 0:
        return []
    s = f"{n:0{width}d}"
    if len(s) != width:
        raise ValueError("Invalid width formatting for digits.")
    return [int(ch) for ch in s]


def _msdd_index(a: int, b: int, digits: int) -> Optional[int]:
    aa = f"{a:0{digits}d}"
    bb = f"{b:0{digits}d}"
    if len(aa) != digits or len(bb) != digits:
        raise ValueError("a or b out of range for requested digits.")
    for k, (da, db) in enumerate(zip(aa, bb)):
        if da != db:
            return k
    return None


def label(a: int, b: int) -> int:
    return 1 if a > b else 0


def question(a: int, b: int) -> str:
    aa = f"{a:06d}"
    bb = f"{b:06d}"
    return f"Is {aa} greater than {bb}?"


def iter_certified_coverage(cfg: Mapping) -> Iterator[SpecExample]:
    """Yield the deterministic certified coverage set.

    MUST:
      - Produce exactly the counts per stratum implied by CoveragePlan.
      - Include `meta['stratum']` for each example.
      - Use canonical formatting for numbers (6 digits, zero-padded).
      - Be stable across machines given the same cfg and seed.

    Stratum S_k generation (MSDD index):
      - Let k be the first digit position where a and b differ (0=most significant).
      - Deterministic scheme (one valid choice; MUST be exactly implemented):
          pairs = [(da, db) for da in 0..9 for db in 0..9 if da != db]  # length 90
          For i in 0..N-1:
              (da, db) = pairs[i % 90]
              prefix = zero_pad(i % (10**k), width=k)  # shared prefix of length k
              suffix_a, suffix_b sampled deterministically from a fixed-seed RNG
              a = prefix + da + suffix_a
              b = prefix + db + suffix_b

    Fail-closed:
      - If any generated pair does not satisfy its intended stratum, abort.
      - If any duplicate prompt is produced, abort.

    """
    wrapper_line, suffix = _require_gate(cfg)
    plan = _require_plan(cfg)

    digits = 6
    if "specs" not in cfg or not isinstance(cfg["specs"], Mapping):
        raise ValueError("cfg['specs'] must be provided.")
    spec_cfg = cfg["specs"].get(SPEC_ID)
    if isinstance(spec_cfg, Mapping):
        a_digits = int(spec_cfg.get("a_digits", digits))
        b_digits = int(spec_cfg.get("b_digits", digits))
        if a_digits != digits or b_digits != digits:
            raise ValueError(f"{SPEC_ID} requires a_digits=b_digits=6 for canonical formatting.")

    pairs: List[Tuple[int, int]] = [(da, db) for da in range(10) for db in range(10) if da != db]
    if len(pairs) != 90:
        raise RuntimeError("Internal error: expected 90 digit pairs.")

    seen_prompts: set[str] = set()

    # S_k strata
    for k in range(digits):
        rng = np.random.default_rng(plan.seed + 1000 * k)
        P = 10**k
        suf_len = digits - (k + 1)
        for i in range(plan.per_msd_stratum_n):
            prefix_i = i % P
            prefix_digits = _digits_base10(prefix_i, k)
            da, db = pairs[i % 90]
            suf_a = rng.integers(0, 10, size=(suf_len,)).tolist()
            suf_b = rng.integers(0, 10, size=(suf_len,)).tolist()
            a_digits_list = prefix_digits + [da] + [int(x) for x in suf_a]
            b_digits_list = prefix_digits + [db] + [int(x) for x in suf_b]
            a = int("".join(str(x) for x in a_digits_list))
            b = int("".join(str(x) for x in b_digits_list))

            msdd = _msdd_index(a, b, digits)
            if msdd != k:
                raise ValueError(f"S_{k} sample failed MSDD check at i={i}: msdd={msdd}")

            a_str = f"{a:06d}"
            b_str = f"{b:06d}"
            prompt = boolqa_prompt(wrapper_line, question(a, b), suffix)
            if prompt in seen_prompts:
                raise ValueError(
                    f"Duplicate prompt detected in certified coverage: {a_str} vs {b_str}"
                )
            seen_prompts.add(prompt)

            yield SpecExample(
                prompt=prompt,
                label=label(a, b),
                meta={
                    "id": f"compare6d/stratum=S_{k}/i={i}",
                    "stratum": f"S_{k}",
                    "k": str(k),
                    "a": a_str,
                    "b": b_str,
                },
            )

    # S_eq stratum
    if plan.eq_n < 0:
        raise ValueError("eq_n must be >= 0")
    for i in range(plan.eq_n):
        a = (i * 1_000_000) // plan.eq_n if plan.eq_n > 0 else 0
        b = a
        a_str = f"{a:06d}"
        b_str = f"{b:06d}"
        prompt = boolqa_prompt(wrapper_line, question(a, b), suffix)
        if prompt in seen_prompts:
            raise ValueError(f"Duplicate prompt detected in certified coverage: {a_str} vs {b_str}")
        seen_prompts.add(prompt)
        if a != b:
            raise ValueError("S_eq generation produced a != b.")
        yield SpecExample(
            prompt=prompt,
            label=label(a, b),
            meta={
                "id": f"compare6d/stratum=S_eq/i={i}",
                "stratum": "S_eq",
                "a": a_str,
                "b": b_str,
            },
        )

    # S_near stratum
    deltas = sorted(set(plan.near_deltas))
    if plan.near_n < 0:
        raise ValueError("near_n must be >= 0")
    if plan.near_n % len(deltas) != 0:
        raise ValueError("near_n must be divisible by len(near_deltas).")
    per_delta_n = plan.near_n // len(deltas)

    for delta in deltas:
        for i in range(per_delta_n):
            base = (i * (1_000_000 - 11)) // per_delta_n if per_delta_n > 0 else 0
            max_base = 999_999 - int(delta)
            if max_base < 0:
                raise ValueError("S_near delta exceeds domain bounds.")
            base = int(base) % (max_base + 1)

            # Enforce global prompt uniqueness across strata deterministically by advancing base on collision.
            # This preserves the S_near constraint (|a-b|=delta) and avoids accidental overlaps with S_k strata.
            for attempt in range(10_000):
                if i % 2 == 1:
                    a, b = base, base + delta  # include negative deltas deterministically
                else:
                    a, b = base + delta, base

                if not (0 <= a <= 999_999 and 0 <= b <= 999_999):
                    raise ValueError("S_near generated out-of-range values.")
                if abs(a - b) != delta:
                    raise ValueError("S_near generated pair does not satisfy intended delta.")

                a_str = f"{a:06d}"
                b_str = f"{b:06d}"
                prompt = boolqa_prompt(wrapper_line, question(a, b), suffix)
                if prompt not in seen_prompts:
                    break
                base = (base + 1) % (max_base + 1)
            else:
                raise ValueError(
                    "Exceeded bounded retries while enforcing prompt uniqueness in S_near."
                )

            seen_prompts.add(prompt)
            yield SpecExample(
                prompt=prompt,
                label=label(a, b),
                meta={
                    "id": f"compare6d/stratum=S_near/d={delta}/i={i}",
                    "stratum": "S_near",
                    "delta": str(delta),
                    "a": a_str,
                    "b": b_str,
                },
            )

    # S_ext stratum
    if plan.ext_n < 0:
        raise ValueError("ext_n must be >= 0")
    if plan.ext_n % 2 != 0:
        raise ValueError("ext_n must be even for symmetric extreme pairing.")

    extremes = [0, 1, 2, 999_997, 999_998, 999_999]
    half = plan.ext_n // 2
    avoid_by_extreme: Dict[int, set[int]] = {}
    for e in extremes:
        avoid = {e}
        for d in deltas:
            avoid.add(max(0, min(999_999, e + d)))
            avoid.add(max(0, min(999_999, e - d)))
        avoid_by_extreme[e] = avoid

    for i in range(half):
        e = extremes[i % len(extremes)]
        avoid = avoid_by_extreme[e]
        other = (i * 9973 + 12345) % 1_000_000
        for _ in range(10_000):
            if other in avoid:
                other = (other + 1) % 1_000_000
                continue
            a = e
            b = other
            a_str = f"{a:06d}"
            b_str = f"{b:06d}"
            prompt = boolqa_prompt(wrapper_line, question(a, b), suffix)
            if prompt not in seen_prompts:
                break
            other = (other + 1) % 1_000_000
        else:
            raise ValueError("Failed to select a unique S_ext pair within bounded retries.")

        seen_prompts.add(prompt)
        yield SpecExample(
            prompt=prompt,
            label=label(a, b),
            meta={
                "id": f"compare6d/stratum=S_ext/i={i}",
                "stratum": "S_ext",
                "a": a_str,
                "b": b_str,
            },
        )

    for i in range(half):
        e = extremes[i % len(extremes)]
        avoid = avoid_by_extreme[e]
        other = (i * 9973 + 12345) % 1_000_000
        for _ in range(10_000):
            if other in avoid:
                other = (other + 1) % 1_000_000
                continue
            a = other
            b = e
            a_str = f"{a:06d}"
            b_str = f"{b:06d}"
            prompt = boolqa_prompt(wrapper_line, question(a, b), suffix)
            if prompt not in seen_prompts:
                break
            other = (other + 1) % 1_000_000
        else:
            raise ValueError("Failed to select a unique S_ext_rev pair within bounded retries.")

        seen_prompts.add(prompt)
        yield SpecExample(
            prompt=prompt,
            label=label(a, b),
            meta={
                "id": f"compare6d/stratum=S_ext_rev/i={i}",
                "stratum": "S_ext",
                "a": a_str,
                "b": b_str,
            },
        )


def iter_additional_search_samples(cfg: Mapping) -> Iterator[SpecExample]:
    """Yield additional bounded search samples beyond certified coverage.

    Used by counterexample search:
      - fixed-seed random interior samples
      - local perturbations around lowest-margin certified points

    This function only defines generation; selection of low-margin points is performed
    in counterexample search logic.

    """
    wrapper_line, suffix = _require_gate(cfg)
    plan = _require_plan(cfg)

    if plan.additional_random_n <= 0:
        return

    rng = np.random.default_rng(plan.seed + 999_999)
    seen_pairs: set[Tuple[int, int]] = set()

    emitted = 0
    attempts = 0
    while emitted < plan.additional_random_n:
        attempts += 1
        if attempts > plan.additional_random_n * 100:
            raise ValueError("Exceeded bounded retries while generating unique additional samples.")

        a = int(rng.integers(0, 1_000_000))
        b = int(rng.integers(0, 1_000_000))
        if a == b:
            continue
        pair = (a, b)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        a_str = f"{a:06d}"
        b_str = f"{b:06d}"
        prompt = boolqa_prompt(wrapper_line, question(a, b), suffix)
        emitted += 1
        yield SpecExample(
            prompt=prompt,
            label=label(a, b),
            meta={
                "id": f"compare6d/search_random/i={emitted - 1}",
                "stratum": "search_random",
                "a": a_str,
                "b": b_str,
            },
        )
