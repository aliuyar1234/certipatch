from __future__ import annotations

import re

import pytest
import torch

from certipatch.eval.metrics import eval_spec_exact
from certipatch.specs import SpecExample

WRAPPER = "Instruction: Answer with a single token: Yes or No."
SUFFIX = "Answer:"


class _FakeTokenizer:
    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        if s == " Yes":
            return [1]
        if s == " No":
            return [2]
        if s == " true":
            return [3]
        if s == " false":
            return [4]
        return [999, 998]

    def decode(self, ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:  # noqa: ARG002
        mapping = {1: " Yes", 2: " No", 3: " true", 4: " false"}
        if len(ids) == 1 and ids[0] in mapping:
            return mapping[ids[0]]
        return "<UNK>"


class _FakeAdapter:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def tokenize(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        idxs: list[int] = []
        for p in prompts:
            m = re.search(r"IDX=(\d+)", p)
            if not m:
                raise ValueError("Prompt missing IDX marker.")
            idxs.append(int(m.group(1)))
        input_ids = torch.tensor(idxs, dtype=torch.int64).unsqueeze(1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        B, T = input_ids.shape
        V = 10
        logits = torch.zeros(B, T, V, dtype=torch.float32)
        idxs = input_ids[:, 0]
        yes = (idxs % 2 == 0).to(dtype=torch.bool)
        logits[yes, 0, 1] = 5.0
        logits[yes, 0, 2] = 0.0
        logits[~yes, 0, 1] = 0.0
        logits[~yes, 0, 2] = 5.0
        return logits


def test_eval_spec_exact_perfect() -> None:
    adapter = _FakeAdapter()
    cfg = {
        "gate": {"enabled": True, "wrapper_line": WRAPPER, "suffix": SUFFIX},
        "answer_tokens": {
            "primary": {"yes": " Yes", "no": " No"},
            "fallback": {"yes": " true", "no": " false"},
        },
    }
    examples = [
        SpecExample(
            prompt=f"{WRAPPER}\nQuestion: IDX={i}\n{SUFFIX}",
            label=1 if i % 2 == 0 else 0,
            meta={},
        )
        for i in range(10)
    ]

    class _NoPatch:  # minimal duck-typed patch
        params: dict = {}

    metrics = eval_spec_exact(cfg=cfg, adapter=adapter, patch=_NoPatch(), examples=examples)  # type: ignore[arg-type]
    assert metrics.total == 10
    assert metrics.failures == 0
    assert metrics.pass_rate == 1.0
    assert metrics.min_margin == 5.0
    assert metrics.p05_margin == 5.0


def test_eval_spec_exact_out_of_scope_prompt_fails_closed() -> None:
    adapter = _FakeAdapter()
    cfg = {
        "gate": {"enabled": True, "wrapper_line": WRAPPER, "suffix": SUFFIX},
        "answer_tokens": {
            "primary": {"yes": " Yes", "no": " No"},
            "fallback": {"yes": " true", "no": " false"},
        },
    }
    bad = SpecExample(prompt=f"{WRAPPER}\nQuestion: IDX=0\nAnswer: x", label=1, meta={})

    class _NoPatch:
        params: dict = {}

    with pytest.raises(ValueError):
        _ = eval_spec_exact(cfg=cfg, adapter=adapter, patch=_NoPatch(), examples=[bad])  # type: ignore[arg-type]
