from __future__ import annotations

from typing import Iterable, Mapping

from certipatch.specs.balance_paren_14 import iter_domain as iter_balance
from certipatch.specs.compare_2d import iter_domain as iter_compare2d
from certipatch.specs.compare_6d_strat import (
    iter_additional_search_samples,
    iter_certified_coverage,
)
from certipatch.specs.parity_4d import iter_domain as iter_parity

WRAPPER = "Instruction: Answer with a single token: Yes or No."
SUFFIX = "Answer:"


def _assert_prompts_well_formed(prompts: Iterable[str]) -> None:
    for p in prompts:
        assert WRAPPER in p.splitlines()
        assert p.rstrip().endswith(SUFFIX)


def test_compare2d_iter_domain_order_and_labels() -> None:
    cfg: Mapping = {
        "gate": {"wrapper_line": WRAPPER, "suffix": SUFFIX},
        "specs": {"compare_2d": {"a_min": 0, "a_max": 1, "b_min": 0, "b_max": 1}},
    }
    ex = list(iter_compare2d(cfg))
    assert len(ex) == 4
    _assert_prompts_well_formed([e.prompt for e in ex])

    assert ex[0].meta["id"] == "compare2d/a=00/b=00"
    assert ex[0].label == 0
    assert ex[1].meta["id"] == "compare2d/a=00/b=01"
    assert ex[1].label == 0
    assert ex[2].meta["id"] == "compare2d/a=01/b=00"
    assert ex[2].label == 1
    assert ex[3].meta["id"] == "compare2d/a=01/b=01"
    assert ex[3].label == 0


def test_parity4d_iter_domain_order_and_labels() -> None:
    cfg: Mapping = {
        "gate": {"wrapper_line": WRAPPER, "suffix": SUFFIX},
        "specs": {"parity_4d": {"n_min": 0, "n_max": 3}},
    }
    ex = list(iter_parity(cfg))
    assert [e.meta["n"] for e in ex] == ["0", "1", "2", "3"]
    assert [e.label for e in ex] == [1, 0, 1, 0]
    _assert_prompts_well_formed([e.prompt for e in ex])


def test_balance14_iter_domain_order_small() -> None:
    cfg: Mapping = {
        "gate": {"wrapper_line": WRAPPER, "suffix": SUFFIX},
        "specs": {"balance_paren_14": {"max_len": 2}},
    }
    ex = list(iter_balance(cfg))
    assert len(ex) == 7
    _assert_prompts_well_formed([e.prompt for e in ex])

    # L=0, b=0 => empty string is balanced.
    assert ex[0].meta["id"] == "balance14/L=0/b=0"
    assert ex[0].label == 1

    # L=1 => "(" then ")"
    assert ex[1].meta["id"] == "balance14/L=1/b=0"
    assert ex[2].meta["id"] == "balance14/L=1/b=1"
    assert ex[1].label == 0
    assert ex[2].label == 0

    # L=2 => "((", "()", ")(", "))"
    assert [e.meta["id"] for e in ex[3:]] == [
        "balance14/L=2/b=0",
        "balance14/L=2/b=1",
        "balance14/L=2/b=2",
        "balance14/L=2/b=3",
    ]
    assert [e.label for e in ex[3:]] == [0, 1, 0, 0]


def _msdd_index(a: int, b: int, digits: int = 6) -> int | None:
    aa = f"{a:0{digits}d}"
    bb = f"{b:0{digits}d}"
    for k, (da, db) in enumerate(zip(aa, bb)):
        if da != db:
            return k
    return None


def test_compare6d_strat_certified_coverage_shapes_and_strata() -> None:
    cfg: Mapping = {
        "gate": {"wrapper_line": WRAPPER, "suffix": SUFFIX},
        "specs": {
            "compare_6d_strat": {
                "a_digits": 6,
                "b_digits": 6,
                "coverage": {
                    # Chosen to deterministically trigger a cross-stratum collision in the naive generator
                    # (S_5 vs S_near) unless uniqueness is enforced.
                    "per_msd_stratum_n": 9360,
                    "eq_n": 0,
                    "near_n": 2500,
                    "ext_n": 2,
                    "seed": 12345,
                    "near_deltas": [1],
                },
            }
        },
    }

    cov = list(iter_certified_coverage(cfg))
    assert len(cov) == (6 * 9360) + 0 + 2500 + 2
    _assert_prompts_well_formed([e.prompt for e in cov])

    prompts = [e.prompt for e in cov]
    assert len(set(prompts)) == len(prompts)

    extremes = {0, 1, 2, 999_997, 999_998, 999_999}
    deltas = {1}
    for e in cov:
        stratum = e.meta["stratum"]
        a = int(e.meta["a"])
        b = int(e.meta["b"])
        if stratum.startswith("S_") and stratum not in {"S_eq", "S_near", "S_ext"}:
            k = int(e.meta["k"])
            assert _msdd_index(a, b, digits=6) == k
        elif stratum == "S_eq":
            assert a == b
        elif stratum == "S_near":
            assert abs(a - b) == int(e.meta["delta"])
            assert int(e.meta["delta"]) in deltas
        elif stratum == "S_ext":
            assert (a in extremes) or (b in extremes)
            assert abs(a - b) not in deltas
        else:
            raise AssertionError(f"Unexpected stratum: {stratum}")


def test_compare6d_strat_additional_search_samples() -> None:
    cfg: Mapping = {
        "gate": {"wrapper_line": WRAPPER, "suffix": SUFFIX},
        "specs": {
            "compare_6d_strat": {
                "a_digits": 6,
                "b_digits": 6,
                "coverage": {
                    "per_msd_stratum_n": 1,
                    "eq_n": 0,
                    "near_n": 2,
                    "ext_n": 2,
                    "seed": 0,
                    "near_deltas": [1, 2],
                    "additional_random_n": 5,
                    "local_perturb_k": 2,
                },
            }
        },
    }

    ex = list(iter_additional_search_samples(cfg))
    assert len(ex) == 5
    _assert_prompts_well_formed([e.prompt for e in ex])
    assert all(int(e.meta["a"]) != int(e.meta["b"]) for e in ex)
