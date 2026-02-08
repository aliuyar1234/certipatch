from __future__ import annotations

from pathlib import Path

import pytest
import torch

from certipatch.cegis import loop as cg
from certipatch.cegis.counterexamples import Counterexample
from certipatch.patch_families import GLRHookPatch, GLRHPConfig
from certipatch.specs import SpecExample


class _FakeAdapter:
    pass


def _cfg(tmp_path: Path, *, run_id: str, resume: bool) -> dict:
    return {
        "run": {"run_id": run_id, "seeds": {"master": 0}},
        "output": {"out_dir": str(tmp_path / "runs")},
        "_certipatch_runtime": {"resume": bool(resume), "refbool_s_prompts": ["r"]},
        "cegis": {
            "max_outer_iters": 3,
            "init_n": {"parity_4d": 2, "compare_2d": 2},
            "k_add": {"parity_4d": 1, "compare_2d": 1},
            "policy": "hardest_margin",
            "inner_solver": "alm",
        },
    }


def _make_patch() -> GLRHookPatch:
    p = GLRHookPatch(GLRHPConfig(rank_r=1, candidate_layers=[0], effective_layer_threshold=0.0))
    p.init_parameters(d_model=2, seed=0)
    return p


def test_run_cegis_resume_from_outer_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, run_id="resume_single", resume=False)
    adapter = _FakeAdapter()
    phase = {"name": "first"}
    solve_calls = {"first": 0, "second": 0}

    monkeypatch.setattr(cg, "_require_refbool_s", lambda _cfg: ["r"])
    monkeypatch.setattr(
        cg,
        "_init_active_set",
        lambda **_kwargs: [SpecExample(prompt="p0", label=1, meta={})],
    )

    def fake_solve_constrained_minimality(*, patch, **_kwargs):  # noqa: ANN003
        key = str(phase["name"])
        solve_calls[key] += 1
        if key == "first" and solve_calls[key] == 2:
            raise RuntimeError("boom-single")
        for layer in patch.cfg.candidate_layers:
            patch.params[layer]["U"] = torch.as_tensor(patch.params[layer]["U"]) + 0.01
        new_state = cg.SolverState(
            lambda_mult=float(solve_calls[key]), mu=1.0, inner_round=int(solve_calls[key])
        )
        return patch, new_state, {"g_true": 0.0}

    def fake_find_counterexamples(*, spec_id, **_kwargs):  # noqa: ANN003
        if phase["name"] == "first":
            if solve_calls["first"] == 1:
                return [
                    Counterexample(
                        prompt=f"cex-{spec_id}",
                        label=1,
                        pred=0,
                        margin=-1.0,
                        meta={"spec_id": str(spec_id)},
                    )
                ]
            return []
        return []

    monkeypatch.setattr(cg, "solve_constrained_minimality", fake_solve_constrained_minimality)
    monkeypatch.setattr(cg, "find_counterexamples", fake_find_counterexamples)

    with pytest.raises(RuntimeError, match="boom-single"):
        _ = cg.run_cegis(cfg=cfg, adapter=adapter, spec_id="parity_4d", patch=_make_patch())

    ckpt_path = tmp_path / "runs" / "resume_single" / "_cegis_ckpt" / "single__parity_4d.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt["outer_next"] == 1
    assert ckpt["done"] is False

    phase["name"] = "second"
    cfg["_certipatch_runtime"]["resume"] = True
    result = cg.run_cegis(cfg=cfg, adapter=adapter, spec_id="parity_4d", patch=_make_patch())
    assert result.outer_iters == 2
    assert len(result.cex_history) == 2
    assert solve_calls["second"] == 1

    ckpt2 = torch.load(ckpt_path, map_location="cpu")
    assert ckpt2["done"] is True
    assert ckpt2["outer_next"] == 2


def test_run_cegis_multi_resume_from_outer_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, run_id="resume_multi", resume=False)
    adapter = _FakeAdapter()
    phase = {"name": "first"}
    solve_calls = {"first": 0, "second": 0}

    monkeypatch.setattr(cg, "_require_refbool_s", lambda _cfg: ["r"])
    monkeypatch.setattr(
        cg,
        "_init_active_set",
        lambda *, spec_id, **_kwargs: [SpecExample(prompt=f"p-{spec_id}", label=1, meta={})],
    )

    def fake_solve_constrained_minimality(*, patch, **_kwargs):  # noqa: ANN003
        key = str(phase["name"])
        solve_calls[key] += 1
        if key == "first" and solve_calls[key] == 2:
            raise RuntimeError("boom-multi")
        for layer in patch.cfg.candidate_layers:
            patch.params[layer]["U"] = torch.as_tensor(patch.params[layer]["U"]) + 0.01
        new_state = cg.SolverState(
            lambda_mult=float(solve_calls[key]), mu=1.0, inner_round=int(solve_calls[key])
        )
        return patch, new_state, {"g_true": 0.0}

    def fake_find_counterexamples(*, spec_id, **_kwargs):  # noqa: ANN003
        if phase["name"] == "first" and solve_calls["first"] == 1 and str(spec_id) == "compare_2d":
            return [
                Counterexample(
                    prompt="cex-compare",
                    label=1,
                    pred=0,
                    margin=-1.0,
                    meta={"spec_id": "compare_2d"},
                )
            ]
        return []

    monkeypatch.setattr(cg, "solve_constrained_minimality", fake_solve_constrained_minimality)
    monkeypatch.setattr(cg, "find_counterexamples", fake_find_counterexamples)

    with pytest.raises(RuntimeError, match="boom-multi"):
        _ = cg._run_cegis_multi(
            cfg=cfg,
            adapter=adapter,
            spec_ids=["compare_2d", "parity_4d"],
            patch=_make_patch(),
            frozen_patch=None,
            checkpoint_key="Joint_AB",
        )

    ckpt_path = tmp_path / "runs" / "resume_multi" / "_cegis_ckpt" / "multi__Joint_AB.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt["outer_next"] == 1
    assert ckpt["done"] is False

    phase["name"] = "second"
    cfg["_certipatch_runtime"]["resume"] = True
    result = cg._run_cegis_multi(
        cfg=cfg,
        adapter=adapter,
        spec_ids=["compare_2d", "parity_4d"],
        patch=_make_patch(),
        frozen_patch=None,
        checkpoint_key="Joint_AB",
    )
    assert result.outer_iters == 2
    assert len(result.cex_history) == 2
    assert solve_calls["second"] == 1

    ckpt2 = torch.load(ckpt_path, map_location="cpu")
    assert ckpt2["done"] is True
    assert ckpt2["outer_next"] == 2


def test_run_cegis_resume_from_inner_solver_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, run_id="resume_inner", resume=False)
    adapter = _FakeAdapter()
    phase = {"name": "first"}
    solve_calls = {"first": 0, "second": 0}
    seen_resume_state = {"inner_round": None}

    monkeypatch.setattr(cg, "_require_refbool_s", lambda _cfg: ["r"])
    monkeypatch.setattr(
        cg,
        "_init_active_set",
        lambda **_kwargs: [SpecExample(prompt="p0", label=1, meta={})],
    )

    def fake_solve_constrained_minimality(*, patch, state=None, on_round_end=None, **_kwargs):  # noqa: ANN003
        key = str(phase["name"])
        solve_calls[key] += 1
        if key == "second":
            seen_resume_state["inner_round"] = None if state is None else int(state.inner_round)
        for layer in patch.cfg.candidate_layers:
            patch.params[layer]["U"] = torch.as_tensor(patch.params[layer]["U"]) + 0.01
        next_round = int((state.inner_round if state is not None else 0) + 1)
        new_state = cg.SolverState(
            lambda_mult=1.0, mu=1.0, inner_round=int(next_round)
        )
        if on_round_end is not None:
            on_round_end(
                patch,
                new_state,
                {"inner_round": int(next_round - 1), "inner_round_next": int(next_round)},
            )
        if key == "first":
            raise RuntimeError("boom-inner")
        return patch, new_state, {"g_true": 0.0}

    monkeypatch.setattr(cg, "solve_constrained_minimality", fake_solve_constrained_minimality)
    monkeypatch.setattr(cg, "find_counterexamples", lambda **_kwargs: [])

    with pytest.raises(RuntimeError, match="boom-inner"):
        _ = cg.run_cegis(cfg=cfg, adapter=adapter, spec_id="parity_4d", patch=_make_patch())

    ckpt_path = tmp_path / "runs" / "resume_inner" / "_cegis_ckpt" / "single__parity_4d.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt["outer_next"] == 0
    assert ckpt["done"] is False
    assert ckpt["state"]["inner_round"] == 1

    phase["name"] = "second"
    cfg["_certipatch_runtime"]["resume"] = True
    result = cg.run_cegis(cfg=cfg, adapter=adapter, spec_id="parity_4d", patch=_make_patch())
    assert result.outer_iters == 1
    assert solve_calls["second"] == 1
    assert seen_resume_state["inner_round"] == 1

    ckpt2 = torch.load(ckpt_path, map_location="cpu")
    assert ckpt2["done"] is True
    assert ckpt2["outer_next"] == 1


def test_run_cegis_multiobjective_resume_from_step_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, run_id="resume_mo_step", resume=False)
    cfg["cegis"]["inner_solver"] = "multiobjective"
    adapter = _FakeAdapter()
    phase = {"name": "first"}
    seen_resume_step = {"value": None}

    monkeypatch.setattr(cg, "_require_refbool_s", lambda _cfg: ["r"])
    monkeypatch.setattr(
        cg,
        "_init_active_set",
        lambda **_kwargs: [SpecExample(prompt="p0", label=1, meta={})],
    )

    def fake_solve_multiobjective(*, patch, resume_step=0, on_step_end=None, **_kwargs):  # noqa: ANN003
        if phase["name"] == "second":
            seen_resume_step["value"] = int(resume_step)
        if on_step_end is not None:
            on_step_end(patch, {"step_next": 25, "steps": 2000, "alpha": 0.1, "loss": 1.0})
        if phase["name"] == "first":
            raise RuntimeError("boom-mo-step")
        return patch, {"g_true": 0.0}

    monkeypatch.setattr(cg, "solve_multiobjective", fake_solve_multiobjective)
    monkeypatch.setattr(cg, "find_counterexamples", lambda **_kwargs: [])

    with pytest.raises(RuntimeError, match="boom-mo-step"):
        _ = cg.run_cegis(cfg=cfg, adapter=adapter, spec_id="parity_4d", patch=_make_patch())

    ckpt_path = tmp_path / "runs" / "resume_mo_step" / "_cegis_ckpt" / "single__parity_4d.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt["outer_next"] == 0
    assert ckpt["done"] is False
    assert ckpt["mo_step_next"] == 25

    phase["name"] = "second"
    cfg["_certipatch_runtime"]["resume"] = True
    result = cg.run_cegis(cfg=cfg, adapter=adapter, spec_id="parity_4d", patch=_make_patch())
    assert result.outer_iters == 1
    assert seen_resume_step["value"] == 25

    ckpt2 = torch.load(ckpt_path, map_location="cpu")
    assert ckpt2["done"] is True
    assert ckpt2["outer_next"] == 1
