from __future__ import annotations

import pytest

from certipatch.models.load_model import _resolve_candidate_layers, assert_or_select_answer_tokens


class _FakeTokenizer:
    def __init__(self, mapping: dict[str, int]) -> None:
        self._mapping = dict(mapping)
        self._reverse = {v: k for k, v in mapping.items()}

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        if s in self._mapping:
            return [self._mapping[s]]
        return [999, 998]

    def decode(self, ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:  # noqa: ARG002
        if len(ids) == 1 and ids[0] in self._reverse:
            return self._reverse[ids[0]]
        return "<UNK>"


class _FakeAdapter:
    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self.tokenizer = tokenizer


def test_resolve_candidate_layers_quartiles() -> None:
    assert _resolve_candidate_layers(12, "quartiles", None) == [3, 6, 9, 11]


def test_resolve_candidate_layers_quartiles_dedup() -> None:
    assert _resolve_candidate_layers(3, "quartiles", None) == [0, 1, 2]


def test_resolve_candidate_layers_explicit() -> None:
    assert _resolve_candidate_layers(5, "explicit", [2, 0, 2]) == [2, 0]


def test_resolve_candidate_layers_out_of_range() -> None:
    with pytest.raises(ValueError):
        _ = _resolve_candidate_layers(3, "explicit", [3])


def test_answer_token_selection_primary() -> None:
    tok = _FakeTokenizer({" Yes": 1, " No": 2, " true": 3, " false": 4})
    adapter = _FakeAdapter(tok)
    cfg = {
        "answer_tokens": {
            "primary": {"yes": " Yes", "no": " No"},
            "fallback": {"yes": " true", "no": " false"},
        }
    }
    sel = assert_or_select_answer_tokens(adapter, cfg)
    assert sel["mode"] == "primary"
    assert sel["yes"] == " Yes"
    assert sel["no"] == " No"


def test_answer_token_selection_fallback() -> None:
    tok = _FakeTokenizer({" true": 3, " false": 4})
    adapter = _FakeAdapter(tok)
    cfg = {
        "answer_tokens": {
            "primary": {"yes": " Yes", "no": " No"},
            "fallback": {"yes": " true", "no": " false"},
        }
    }
    sel = assert_or_select_answer_tokens(adapter, cfg)
    assert sel["mode"] == "fallback"
    assert sel["yes"] == " true"
    assert sel["no"] == " false"
