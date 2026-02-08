from __future__ import annotations

import torch

from certipatch.patch_families import GLRHookPatch, GLRHPConfig


def test_glrhp_init_parameters_is_deterministic() -> None:
    cfg = GLRHPConfig(rank_r=2, candidate_layers=[0, 3], effective_layer_threshold=0.0)
    p1 = GLRHookPatch(cfg)
    p2 = GLRHookPatch(cfg)
    p1.init_parameters(d_model=4, seed=123)
    p2.init_parameters(d_model=4, seed=123)

    for layer in cfg.candidate_layers:
        assert torch.allclose(p1.params[layer]["U"], p2.params[layer]["U"])
        assert torch.allclose(p1.params[layer]["V"], p2.params[layer]["V"])


def test_glrhp_apply_to_vectors_matches_formula() -> None:
    torch.manual_seed(0)
    cfg = GLRHPConfig(rank_r=3, candidate_layers=[1], effective_layer_threshold=0.0)
    patch = GLRHookPatch(cfg)
    patch.init_parameters(d_model=5, seed=0)

    h = torch.randn(7, 5)
    U = patch.params[1]["U"]
    V = patch.params[1]["V"]
    expected = h + (h @ V) @ U.T
    actual = patch.apply_to_vectors(h, layer=1)
    assert torch.allclose(actual, expected)


def test_glrhp_effective_layers_threshold() -> None:
    cfg = GLRHPConfig(rank_r=2, candidate_layers=[0], effective_layer_threshold=1e9)
    patch = GLRHookPatch(cfg)
    patch.init_parameters(d_model=4, seed=0)
    assert patch.effective_layers() == []
