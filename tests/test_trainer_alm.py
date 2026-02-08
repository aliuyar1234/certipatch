from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from certipatch.cegis.trainer import SolverState, solve_constrained_minimality
from certipatch.models.load_model import ModelInfo
from certipatch.patch_families import GLRHookPatch, GLRHPConfig
from certipatch.specs import SpecExample

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


@dataclass
class _Handle:
    hooks: list[Callable[[Any, Any, Any], Any]]
    fn: Callable[[Any, Any, Any], Any]

    def remove(self) -> None:
        self.hooks.remove(self.fn)


class _FakeAdapter:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.info = ModelInfo(
            backend="huggingface",
            model_path_or_id="fake",
            revision="fake",
            tokenizer_path_or_id="fake",
            d_model=1,
            n_layers=1,
        )
        self._hooks: Dict[Tuple[str, int], List[Callable[[Any, Any, Any], Any]]] = {}

    def tokenize(self, prompts: Sequence[str]) -> Dict[str, Any]:
        ys: list[int] = []
        for p in prompts:
            m = re.search(r"Y=(\d+)", p)
            if not m:
                raise ValueError("Prompt missing Y marker.")
            ys.append(int(m.group(1)))
        input_ids = torch.tensor(ys, dtype=torch.int64).unsqueeze(1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward_logits(self, input_ids: Any, attention_mask: Any) -> Any:  # noqa: ARG002
        ids = torch.as_tensor(input_ids, dtype=torch.int64)
        B, T = ids.shape
        d_model = 1

        # Base hidden state encodes "label direction" with small magnitude, so tau=1 requires a small update.
        y = ids[:, 0]
        h = torch.zeros(B, T, d_model, dtype=torch.float32)
        h[y == 1, :, 0] = 0.49
        h[y == 0, :, 0] = -0.49
        h[y == 2, :, 0] = 0.0  # reference prompts (no collateral)

        for fn in self._hooks.get(("resid_post", 0), []):
            h = fn(h, None, None)

        V = 5
        logits = torch.zeros(B, T, V, dtype=torch.float32)
        logits[:, :, 1] = h[:, :, 0]
        logits[:, :, 2] = -h[:, :, 0]
        return logits

    def resolve_candidate_layers(
        self, mode: str, explicit: Optional[List[int]] = None
    ) -> List[int]:  # noqa: ARG002
        if explicit is not None:
            return [int(x) for x in explicit]
        return [0]

    def register_hook(
        self,
        kind: str,
        layer: int,
        hook_fn: Callable[[Any, Any, Any], Any],
    ) -> Any:
        key = (kind, int(layer))
        hooks = self._hooks.setdefault(key, [])
        hooks.append(hook_fn)
        return _Handle(hooks=hooks, fn=hook_fn)


def test_solve_constrained_minimality_reaches_feasible() -> None:
    cfg: Mapping[str, Any] = {
        "run": {"seeds": {"torch": 0, "master": 0}},
        "gate": {"enabled": True, "wrapper_line": WRAPPER, "suffix": SUFFIX},
        "answer_tokens": {
            "primary": {"yes": " Yes", "no": " No"},
            "fallback": {"yes": " true", "no": " false"},
        },
        "hookpoints": {
            "kind": "resid_post",
            "candidate_layers": {"mode": "explicit", "explicit": [0]},
        },
        "patch": {"family": "GLR-HP", "rank_r": 1, "effective_layer_threshold": 0.0},
        "objective": {"tau_margin": 1.0, "beta_smooth": 50.0},
        "regularizers": {"lambda_l2": 0.0, "lambda_group": 0.0},
        "optimizer": {
            "lr": 0.5,
            "adam_betas": [0.9, 0.999],
            "inner_steps_per_outer": 200,
            "max_inner_rounds": 3,
            "patience_steps": 10_000,
            "spec_batch_size": 8,
            "ref_batch_size": 8,
        },
        "alm": {
            "mu_init": 1.0,
            "mu_mult_on_violation": 10.0,
            "mu_div_on_feasible": 2.0,
            "mu_floor": 0.001,
        },
    }

    adapter = _FakeAdapter()
    patch_cfg = GLRHPConfig(rank_r=1, candidate_layers=[0], effective_layer_threshold=0.0)
    patch = GLRHookPatch(patch_cfg)
    patch.init_parameters(d_model=1, seed=0)

    D_spec = [
        SpecExample(prompt=f"{WRAPPER}\nQuestion: Y={y}\n{SUFFIX}", label=y, meta={})
        for y in [1, 0, 1, 0, 1, 0, 1, 0]
    ]
    D_ref = [f"{WRAPPER}\nQuestion: Y=2\n{SUFFIX}" for _ in range(8)]

    patch2, state, diag = solve_constrained_minimality(
        cfg=cfg,
        adapter=adapter,  # type: ignore[arg-type]
        patch=patch,
        D_spec=D_spec,
        D_ref=D_ref,
    )

    assert patch2 is patch
    assert diag["g_true"] == 0.0
    assert state.inner_round >= 1


def test_solve_constrained_minimality_does_not_skip_when_state_inner_round_at_limit() -> None:
    cfg: Mapping[str, Any] = {
        "run": {"seeds": {"torch": 0, "master": 0}},
        "gate": {"enabled": True, "wrapper_line": WRAPPER, "suffix": SUFFIX},
        "answer_tokens": {
            "primary": {"yes": " Yes", "no": " No"},
            "fallback": {"yes": " true", "no": " false"},
        },
        "hookpoints": {
            "kind": "resid_post",
            "candidate_layers": {"mode": "explicit", "explicit": [0]},
        },
        "patch": {"family": "GLR-HP", "rank_r": 1, "effective_layer_threshold": 0.0},
        "objective": {"tau_margin": 1.0, "beta_smooth": 50.0},
        "regularizers": {"lambda_l2": 0.0, "lambda_group": 0.0},
        "optimizer": {
            "lr": 0.5,
            "adam_betas": [0.9, 0.999],
            "inner_steps_per_outer": 10,
            "max_inner_rounds": 1,
            "patience_steps": 0,
            "spec_batch_size": 8,
            "ref_batch_size": 8,
        },
        "alm": {
            "mu_init": 1.0,
            "mu_mult_on_violation": 10.0,
            "mu_div_on_feasible": 2.0,
            "mu_floor": 0.001,
        },
    }

    adapter = _FakeAdapter()
    patch_cfg = GLRHPConfig(rank_r=1, candidate_layers=[0], effective_layer_threshold=0.0)
    patch = GLRHookPatch(patch_cfg)
    patch.init_parameters(d_model=1, seed=0)

    D_spec = [
        SpecExample(prompt=f"{WRAPPER}\nQuestion: Y={y}\n{SUFFIX}", label=y, meta={})
        for y in [1, 0, 1, 0, 1, 0, 1, 0]
    ]
    D_ref = [f"{WRAPPER}\nQuestion: Y=2\n{SUFFIX}" for _ in range(8)]

    _patch1, state1, _diag1 = solve_constrained_minimality(
        cfg=cfg,
        adapter=adapter,  # type: ignore[arg-type]
        patch=patch,
        D_spec=D_spec,
        D_ref=D_ref,
    )

    # Regression: if state.inner_round is at/above max_inner_rounds, the solver must still run (not skip).
    state_bad = SolverState(
        lambda_mult=float(state1.lambda_mult), mu=float(state1.mu), inner_round=1
    )
    _patch2, state2, diag2 = solve_constrained_minimality(
        cfg=cfg,
        adapter=adapter,  # type: ignore[arg-type]
        patch=patch,
        D_spec=D_spec,
        D_ref=D_ref,
        state=state_bad,
    )

    assert diag2["g_true"] is not None
    assert state2.inner_round > state_bad.inner_round


def test_solve_constrained_minimality_sanitizes_extreme_resume_state() -> None:
    cfg: Mapping[str, Any] = {
        "run": {"seeds": {"torch": 0, "master": 0}},
        "gate": {"enabled": True, "wrapper_line": WRAPPER, "suffix": SUFFIX},
        "answer_tokens": {
            "primary": {"yes": " Yes", "no": " No"},
            "fallback": {"yes": " true", "no": " false"},
        },
        "hookpoints": {
            "kind": "resid_post",
            "candidate_layers": {"mode": "explicit", "explicit": [0]},
        },
        "patch": {"family": "GLR-HP", "rank_r": 1, "effective_layer_threshold": 0.0},
        "objective": {"tau_margin": 1.0, "beta_smooth": 50.0},
        "regularizers": {"lambda_l2": 0.0, "lambda_group": 0.0},
        "optimizer": {
            "lr": 0.5,
            "adam_betas": [0.9, 0.999],
            "inner_steps_per_outer": 5,
            "max_inner_rounds": 1,
            "patience_steps": 0,
            "spec_batch_size": 8,
            "ref_batch_size": 8,
            "grad_clip_norm": 1.0,
        },
        "alm": {
            "mu_init": 1.0,
            "mu_mult_on_violation": 10.0,
            "mu_div_on_feasible": 2.0,
            "mu_floor": 0.001,
            "mu_ceiling": 10.0,
            "lambda_ceiling": 10.0,
            "non_finite_backoff_factor": 10.0,
            "non_finite_max_backoffs": 8,
        },
    }

    adapter = _FakeAdapter()
    patch_cfg = GLRHPConfig(rank_r=1, candidate_layers=[0], effective_layer_threshold=0.0)
    patch = GLRHookPatch(patch_cfg)
    patch.init_parameters(d_model=1, seed=0)

    D_spec = [
        SpecExample(prompt=f"{WRAPPER}\nQuestion: Y={y}\n{SUFFIX}", label=y, meta={})
        for y in [1, 0, 1, 0, 1, 0, 1, 0]
    ]
    D_ref = [f"{WRAPPER}\nQuestion: Y=2\n{SUFFIX}" for _ in range(8)]

    extreme_state = SolverState(lambda_mult=float("inf"), mu=float("inf"), inner_round=0)
    _patch_out, state_out, diag_out = solve_constrained_minimality(
        cfg=cfg,
        adapter=adapter,  # type: ignore[arg-type]
        patch=patch,
        D_spec=D_spec,
        D_ref=D_ref,
        state=extreme_state,
    )

    assert math.isfinite(float(state_out.mu))
    assert math.isfinite(float(state_out.lambda_mult))
    assert abs(float(state_out.lambda_mult)) <= 10.0
    assert float(state_out.mu) <= 10.0
    assert math.isfinite(float(diag_out["lambda_mult"]))
    assert math.isfinite(float(diag_out["mu"]))
