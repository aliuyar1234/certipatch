"""certipatch.data_generation

Deterministic offline generators for collateral suites and hashing utilities.

SSOT references:
- DATA_GENERATION.md
- 02_Specs_Domains.md
- 08_Collateral_Suites.md

These helpers are used by:
- scripts/reproduce_paper.py (to build suites and hashes)
- certipatch.artifacts.verifier (to recompute and verify hashes)

Fail-closed:
- If wordlists are missing or a generated prompt violates gating expectations, abort.
- If a collateral prompt overlaps a spec prompt by exact string match, abort.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from certipatch.hooks import GateSpec, boolqa_gate
from certipatch.specs import SpecExample, boolqa_prompt


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def domain_hash(examples: Iterable[SpecExample]) -> str:
    """Hash canonical prompts+labels in order: '{prompt}\\t{label}\\n'."""
    h = sha256()
    for e in examples:
        h.update(e.prompt.encode("utf-8"))
        h.update(b"\t")
        h.update(str(int(e.label)).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def suite_hash(prompts: Sequence[str], *, extra: Optional[Mapping[str, Any]] = None) -> str:
    """Hash suite prompts; include extra config in the hash input when provided."""
    h = sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    if extra is not None:
        blob = json.dumps(extra, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        h.update(b"\n--extra--\n")
        h.update(blob)
        h.update(b"\n")
    return h.hexdigest()


def _read_wordlist(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Wordlist not found: {path.as_posix()}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    out = [ln for ln in lines if ln]
    if not out:
        raise ValueError(f"Wordlist is empty: {path.as_posix()}")
    return out


def _wordlists(cfg: Mapping[str, Any]) -> dict[str, list[str]]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), Mapping) else {}
    wordlists_dir = Path(str(data_cfg.get("wordlists_dir", "assets/wordlists"))).resolve()
    return {
        "animals": _read_wordlist(wordlists_dir / "animals.txt"),
        "cities": _read_wordlist(wordlists_dir / "cities.txt"),
        "colors": _read_wordlist(wordlists_dir / "colors.txt"),
        "months": _read_wordlist(wordlists_dir / "months.txt"),
        "substrings": _read_wordlist(wordlists_dir / "substrings.txt"),
        "words": _read_wordlist(wordlists_dir / "words.txt"),
    }


def _require_gate_cfg(cfg: Mapping[str, Any]) -> tuple[str, str, GateSpec]:
    gate_cfg = cfg.get("gate", {})
    if not isinstance(gate_cfg, Mapping):
        raise ValueError("cfg['gate'] must be a mapping.")
    wrapper_line = str(gate_cfg["wrapper_line"])
    suffix = str(gate_cfg["suffix"])
    return wrapper_line, suffix, GateSpec(wrapper_line=wrapper_line, suffix=suffix)


def build_refbool_s(
    *,
    cfg: Mapping[str, Any],
    n_prompts: int,
    spec_prompt_set: set[str],
) -> list[str]:
    """Deterministic RefBool-S generator (gate=1, disjoint from spec prompts)."""
    n = int(n_prompts)
    if n <= 0:
        raise ValueError("n_prompts must be > 0 for RefBool-S.")
    wrapper_line, suffix, gate = _require_gate_cfg(cfg)
    wl = _wordlists(cfg)

    words = wl["words"]
    substrings = wl["substrings"]
    months = wl["months"]
    colors = wl["colors"]
    animals = wl["animals"]
    cities = wl["cities"]

    def q_len_parity(w: str) -> str:
        return f"Is the length of the string '{w}' even?"

    def q_vowel_count(w: str) -> str:
        return f"Does the word '{w}' contain more than 2 vowels?"

    def q_alpha_order(w1: str, w2: str) -> str:
        return f"Is '{w1}' alphabetically before '{w2}'?"

    def q_contains_sub(w: str, sub: str) -> str:
        return f"Does '{w}' contain the substring '{sub}'?"

    def q_month_half(m: str) -> str:
        return f"Is '{m}' in the first half of the year?"

    def q_primary_color(c: str) -> str:
        return f"Is '{c}' a primary color?"

    def q_all_lower(w: str) -> str:
        return f"Is '{w}' all lowercase?"

    def q_palindrome(w: str) -> str:
        return f"Is '{w}' a palindrome?"

    def q_mammal(a: str) -> str:
        return f"Is '{a}' a mammal?"

    def q_europe(city: str) -> str:
        return f"Is '{city}' in Europe?"

    families: list[tuple[str, Iterable[str]]] = [
        ("len_parity", (q_len_parity(w) for w in words)),
        ("vowels_gt2", (q_vowel_count(w) for w in words)),
        (
            "alpha_order",
            (
                q_alpha_order(words[i % len(words)], words[(i + 1) % len(words)])
                for i in range(len(words))
            ),
        ),
        (
            "contains_sub",
            (
                q_contains_sub(words[i % len(words)], substrings[i % len(substrings)])
                for i in range(max(len(words), len(substrings)))
            ),
        ),
        ("month_half", (q_month_half(m) for m in months)),
        ("primary_color", (q_primary_color(c) for c in colors)),
        ("all_lower", (q_all_lower(w) for w in words)),
        ("palindrome", (q_palindrome(w) for w in words)),
        ("mammal", (q_mammal(a) for a in animals)),
        ("europe", (q_europe(c) for c in cities)),
    ]

    out: list[str] = []
    for _family_name, questions in families:
        for q in questions:
            prompt = boolqa_prompt(wrapper_line, q, suffix)
            if prompt in spec_prompt_set:
                raise ValueError("RefBool-S prompt overlaps spec prompt set; adjust templates.")
            if not boolqa_gate(prompt, gate):
                raise ValueError(
                    "RefBool-S generator produced an out-of-scope prompt (gate=false)."
                )
            out.append(prompt)
            if len(out) >= n:
                return out

    # If we didn't fill, extend deterministically by cycling through words with a seeded RNG.
    rng = np.random.default_rng(0)
    while len(out) < n:
        w = words[int(rng.integers(0, len(words)))]
        q = q_len_parity(w)
        prompt = boolqa_prompt(wrapper_line, q, suffix)
        if prompt in spec_prompt_set:
            continue
        if not boolqa_gate(prompt, gate):
            raise ValueError("RefBool-S generator produced an out-of-scope prompt (gate=false).")
        out.append(prompt)
    return out


def build_refbool_l(
    *,
    cfg: Mapping[str, Any],
    n_prompts: int,
    spec_prompt_set: set[str],
) -> list[str]:
    """Deterministic RefBool-L generator (gate=1, asks for one-sentence explanation)."""
    n = int(n_prompts)
    if n <= 0:
        raise ValueError("n_prompts must be > 0 for RefBool-L.")
    wrapper_line, suffix, gate = _require_gate_cfg(cfg)
    wl = _wordlists(cfg)

    animals = wl["animals"]
    cities = wl["cities"]
    colors = wl["colors"]
    months = wl["months"]

    def q(p: str) -> str:
        return f"{p} Also give one sentence explaining your answer after the Yes/No."

    base_questions = [
        *(f"Is a {a} an animal?" for a in animals),
        *(f"Is {c} a city?" for c in cities),
        *(f"Is {c} a color?" for c in colors),
        *(f"Is {m} a month of the year?" for m in months),
    ]

    out: list[str] = []
    for b in base_questions:
        prompt = boolqa_prompt(wrapper_line, q(b), suffix)
        if prompt in spec_prompt_set:
            raise ValueError("RefBool-L prompt overlaps spec prompt set; adjust templates.")
        if not boolqa_gate(prompt, gate):
            raise ValueError("RefBool-L generator produced an out-of-scope prompt (gate=false).")
        out.append(prompt)
        if len(out) >= n:
            return out

    rng = np.random.default_rng(1)
    while len(out) < n:
        a = animals[int(rng.integers(0, len(animals)))]
        prompt = boolqa_prompt(wrapper_line, q(f"Is a {a} an animal?"), suffix)
        if prompt in spec_prompt_set:
            continue
        if not boolqa_gate(prompt, gate):
            raise ValueError("RefBool-L generator produced an out-of-scope prompt (gate=false).")
        out.append(prompt)
    return out


def build_reftext(
    *,
    cfg: Mapping[str, Any],
    n_prompts: int,
) -> list[str]:
    """Deterministic gate-off RefText generator (gate=0)."""
    n = int(n_prompts)
    if n <= 0:
        raise ValueError("n_prompts must be > 0 for RefText.")
    _, _, gate = _require_gate_cfg(cfg)
    wl = _wordlists(cfg)

    animals = wl["animals"]
    colors = wl["colors"]
    words = wl["words"]

    out: list[str] = []
    for i in range(n):
        a = animals[i % len(animals)]
        c = colors[i % len(colors)]
        w = words[i % len(words)]
        prompt = f"The {a} saw a {c} {w}."
        if boolqa_gate(prompt, gate):
            raise ValueError(
                "RefText generator produced an in-scope prompt (gate=true); adjust template."
            )
        out.append(prompt)
    return out


def coverage_plan_dict(cfg: Mapping[str, Any], *, generator_code_hash: str) -> dict[str, Any]:
    """Return a canonical dict describing the compare_6d_strat certified coverage plan."""
    specs = cfg.get("specs", {}) if isinstance(cfg.get("specs"), Mapping) else {}
    spec_cfg = (
        specs.get("compare_6d_strat", {})
        if isinstance(specs.get("compare_6d_strat"), Mapping)
        else {}
    )
    coverage = spec_cfg.get("coverage", {}) if isinstance(spec_cfg.get("coverage"), Mapping) else {}

    return {
        "spec_id": "compare_6d_strat",
        "generator_code_hash": str(generator_code_hash),
        "prng": {
            "name": "numpy.default_rng",
            "algorithm": "PCG64",
            "numpy_version": str(np.__version__),
        },
        "counts": {
            "per_msd_stratum_n": int(coverage.get("per_msd_stratum_n", 0)),
            "eq_n": int(coverage.get("eq_n", 0)),
            "near_n": int(coverage.get("near_n", 0)),
            "ext_n": int(coverage.get("ext_n", 0)),
        },
        "seed": int(coverage.get("seed", 0)),
        "near_deltas": list(coverage.get("near_deltas", [])),
        "ordering": "strata_then_index",
    }


def coverage_plan_hash(cfg: Mapping[str, Any], *, generator_code_hash: str) -> str:
    plan = coverage_plan_dict(cfg, generator_code_hash=generator_code_hash)
    blob = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return _sha256_bytes(blob)
