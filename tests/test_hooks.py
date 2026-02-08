from __future__ import annotations

import pytest
import torch

from certipatch.hooks import GateSpec, answer_positions, boolqa_gate, gather_logits_at_positions

WRAPPER = "Instruction: Answer with a single token: Yes or No."
SUFFIX = "Answer:"


def test_boolqa_gate_true_with_exact_wrapper_and_suffix() -> None:
    gate = GateSpec(wrapper_line=WRAPPER, suffix=SUFFIX)
    prompt = f"{WRAPPER}\nQuestion: Is 00 greater than 00?\n{SUFFIX}\n"
    assert boolqa_gate(prompt, gate) is True


def test_boolqa_gate_false_if_missing_wrapper_line() -> None:
    gate = GateSpec(wrapper_line=WRAPPER, suffix=SUFFIX)
    prompt = f"Instruction: something else\nQuestion: x\n{SUFFIX}\n"
    assert boolqa_gate(prompt, gate) is False


def test_boolqa_gate_false_if_missing_suffix_line() -> None:
    gate = GateSpec(wrapper_line=WRAPPER, suffix=SUFFIX)
    prompt = f"{WRAPPER}\nQuestion: x\nAnswer: x\n"
    assert boolqa_gate(prompt, gate) is False


def test_answer_positions_sum_minus_one() -> None:
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int64)
    p = answer_positions(mask)
    assert p.tolist() == [2, 1]


def test_answer_positions_fail_on_zero_sum() -> None:
    mask = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    with pytest.raises(ValueError):
        _ = answer_positions(mask)


def test_gather_logits_at_positions() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    p = torch.tensor([0, 2], dtype=torch.int64)
    gathered = gather_logits_at_positions(logits, p)
    assert gathered.shape == (2, 5)
    assert torch.allclose(gathered[0], logits[0, 0])
    assert torch.allclose(gathered[1], logits[1, 2])
