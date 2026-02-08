from __future__ import annotations

import re

import torch

from certipatch.cegis.counterexamples import Counterexample, find_counterexamples, select_hardest

WRAPPER = "Instruction: Answer with a single token: Yes or No."
SUFFIX = "Answer:"


class _FakeTokenizer:
    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        if s == " Yes":
            return [1]
        if s == " No":
            return [2]
        return [999, 998]

    def decode(self, ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:  # noqa: ARG002
        if ids == [1]:
            return " Yes"
        if ids == [2]:
            return " No"
        return "<UNK>"


class _FakeAdapter:
    def __init__(self, *, invert: bool) -> None:
        self.tokenizer = _FakeTokenizer()
        self._invert = invert

    def tokenize(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        a_vals: list[int] = []
        b_vals: list[int] = []
        for p in prompts:
            m = re.search(r"Is (\d{2}) greater than (\d{2})\?", p)
            if not m:
                raise ValueError("Prompt did not match expected compare_2d format.")
            a_vals.append(int(m.group(1)))
            b_vals.append(int(m.group(2)))
        input_ids = torch.tensor(list(zip(a_vals, b_vals, strict=True)), dtype=torch.int64)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        B, T = input_ids.shape
        V = 10
        logits = torch.zeros(B, T, V, dtype=torch.float32)
        a = input_ids[:, 0]
        b = input_ids[:, 1]
        pred_yes = a > b
        if self._invert:
            pred_yes = ~pred_yes
        # Write answer-position logits at the last non-pad token (here: position 1).
        logits[pred_yes, 1, 1] = 5.0
        logits[pred_yes, 1, 2] = 0.0
        logits[~pred_yes, 1, 1] = 0.0
        logits[~pred_yes, 1, 2] = 5.0
        return logits


class _NoPatch:
    params: dict = {}


def test_select_hardest_tiebreak() -> None:
    cex = [
        Counterexample(prompt="b", label=0, pred=1, margin=0.1, meta={}),
        Counterexample(prompt="c", label=0, pred=1, margin=-0.5, meta={}),
        Counterexample(prompt="a", label=0, pred=1, margin=-0.5, meta={}),
    ]
    out = select_hardest(cex, k_add=2)
    assert [c.prompt for c in out] == ["a", "c"]


def test_find_counterexamples_compare2d_all_failures_when_inverted() -> None:
    cfg = {
        "gate": {"enabled": True, "wrapper_line": WRAPPER, "suffix": SUFFIX},
        "answer_tokens": {
            "primary": {"yes": " Yes", "no": " No"},
            "fallback": {"yes": " true", "no": " false"},
        },
        "specs": {"compare_2d": {"a_min": 0, "a_max": 1, "b_min": 0, "b_max": 1}},
    }
    adapter = _FakeAdapter(invert=True)

    cex = find_counterexamples(cfg=cfg, adapter=adapter, patch=_NoPatch(), spec_id="compare_2d")  # type: ignore[arg-type]
    assert len(cex) == 4
    assert all(c.margin == -5.0 for c in cex)
    assert all(c.meta["spec_id"] == "compare_2d" for c in cex)
    assert cex[0].meta["id"] == "compare2d/a=00/b=00"


def test_find_counterexamples_compare2d_no_failures_when_correct() -> None:
    cfg = {
        "gate": {"enabled": True, "wrapper_line": WRAPPER, "suffix": SUFFIX},
        "answer_tokens": {
            "primary": {"yes": " Yes", "no": " No"},
            "fallback": {"yes": " true", "no": " false"},
        },
        "specs": {"compare_2d": {"a_min": 0, "a_max": 1, "b_min": 0, "b_max": 1}},
    }
    adapter = _FakeAdapter(invert=False)
    cex = find_counterexamples(cfg=cfg, adapter=adapter, patch=_NoPatch(), spec_id="compare_2d")  # type: ignore[arg-type]
    assert cex == []
