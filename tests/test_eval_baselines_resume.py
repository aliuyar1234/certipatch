from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from certipatch.eval import baselines as bl
from certipatch.patch_families import GLRHookPatch, GLRHPConfig
from certipatch.specs import SpecExample


class _FakeAdapter:
    def __init__(self) -> None:
        self.info = SimpleNamespace(backend="huggingface", d_model=16, n_layers=4)


class _FakePatch:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def parameter_count(self) -> int:
        return 10


class _FakeSteeringPatch:
    params: dict = {}

    def parameter_count(self) -> int:
        return 12

    def serialize(self) -> dict:
        return {"family": "SteeringVec-1L", "parameter_count": 12}


def _tiny_patch() -> GLRHookPatch:
    patch = GLRHookPatch(GLRHPConfig(rank_r=1, candidate_layers=[0], effective_layer_threshold=0.0))
    patch.init_parameters(d_model=4, seed=0)
    return patch


def _base_cfg(tmp_path) -> dict:
    out_dir = tmp_path / "runs"
    return {
        "run": {"run_id": "resume_test"},
        "output": {"out_dir": str(out_dir)},
        "_certipatch_runtime": {
            "resume": False,
            "refbool_s_prompts": ["S"],
            "refbool_l_prompts": ["L"],
            "reftext_prompts": ["T"],
        },
        "baselines": {"enabled": ["oneshot_full_mo"]},
        "baseline": {"alpha_grid": [0.0, 0.1, 0.2]},
        "specs": {"enabled": ["parity_4d"]},
        "objective": {"tau_margin": 1.0},
        "patch": {"rank_r": 4},
    }


def test_run_baselines_resume_recovers_from_partial_alpha_grid(tmp_path, monkeypatch) -> None:
    cfg = _base_cfg(tmp_path)
    adapter = _FakeAdapter()
    calls: list[float] = []
    phase = {"name": "first"}

    def fake_iter_domain_examples(_cfg, _spec_id):  # noqa: ANN001
        return [SpecExample(prompt="p0", label=1, meta={})]

    def fake_make_glr_patch(_cfg, *, cand_layers):  # noqa: ANN001
        assert cand_layers == [0]
        return _FakePatch(alpha=-1.0)

    def fake_solve_multiobjective(*, alpha, **_kwargs):  # noqa: ANN003
        alpha = float(alpha)
        calls.append(alpha)
        if phase["name"] == "first" and alpha == pytest.approx(0.1):
            raise RuntimeError("simulated crash during alpha grid")
        return _FakePatch(alpha=alpha), {"alpha": alpha}

    def fake_eval_spec_exact(*, patch, **_kwargs):  # noqa: ANN003
        assert isinstance(patch, _FakePatch)
        return SimpleNamespace(failures=0, min_margin=1.5)

    def fake_eval_collateral(*, patch, **_kwargs):  # noqa: ANN003
        assert isinstance(patch, _FakePatch)
        return SimpleNamespace(refbool_s_mean_kl=float(1.0 - patch.alpha))

    def fake_eval_all_specs_glr(*, patch, **_kwargs):  # noqa: ANN003
        assert isinstance(patch, _FakePatch)
        return {
            "spec_metrics": {"parity_4d": {"failures": 0, "min_margin": 1.5}},
            "collateral_metrics": {"refbool_s_mean_kl": float(1.0 - patch.alpha)},
        }

    monkeypatch.setattr(bl, "_resolve_candidate_layers", lambda _cfg, _adapter: [0])
    monkeypatch.setattr(
        bl,
        "_glrhp_budget",
        lambda _adapter, **_kwargs: {"P_GLRHP": 10, "budget_lo": 9, "budget_hi": 11},
    )
    monkeypatch.setattr(bl, "_iter_domain_examples", fake_iter_domain_examples)
    monkeypatch.setattr(bl, "_make_glr_patch", fake_make_glr_patch)
    monkeypatch.setattr(bl, "solve_multiobjective", fake_solve_multiobjective)
    monkeypatch.setattr(bl, "eval_spec_exact", fake_eval_spec_exact)
    monkeypatch.setattr(bl, "eval_collateral", fake_eval_collateral)
    monkeypatch.setattr(bl, "_eval_all_specs_glr", fake_eval_all_specs_glr)

    with pytest.raises(RuntimeError, match="simulated crash"):
        _ = bl.run_baselines(cfg=cfg, adapter=adapter)

    ckpt_root = tmp_path / "runs" / "resume_test" / "_baselines_ckpt"
    alpha0_ckpt = ckpt_root / "oneshot_full_mo" / f"alpha_{bl._alpha_tag(0.0)}.json"
    alpha01_ckpt = ckpt_root / "oneshot_full_mo" / f"alpha_{bl._alpha_tag(0.1)}.json"
    assert alpha0_ckpt.exists()
    assert not alpha01_ckpt.exists()
    assert calls == [0.0, 0.1]

    phase["name"] = "second"
    calls.clear()
    cfg["_certipatch_runtime"]["resume"] = True
    results = bl.run_baselines(cfg=cfg, adapter=adapter)
    assert len(results) == 1
    assert results[0].name == "oneshot_full_mo"
    assert results[0].metrics["alpha_search"]["selected"] == 0.2
    assert calls == [0.1, 0.2]
    assert (ckpt_root / "oneshot_full_mo.json").exists()

    phase["name"] = "third"
    calls.clear()
    results2 = bl.run_baselines(cfg=cfg, adapter=adapter)
    assert len(results2) == 1
    assert calls == []


def test_run_baselines_rebuilds_missing_prompt_suites(tmp_path, monkeypatch) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["baselines"] = {"enabled": ["steering_vec_1l"]}
    cfg["_certipatch_runtime"] = {"resume": True}
    adapter = _FakeAdapter()
    calls = {"ref_s": 0, "ref_l": 0, "ref_t": 0}

    def fake_iter_domain_examples(_cfg, _spec_id):  # noqa: ANN001
        return [SpecExample(prompt="domain_prompt", label=1, meta={})]

    def fake_build_refbool_s(*, cfg, n_prompts, spec_prompt_set):  # noqa: ANN001
        assert n_prompts > 0
        assert "domain_prompt" in spec_prompt_set
        calls["ref_s"] += 1
        return ["S0", "S1"]

    def fake_build_refbool_l(*, cfg, n_prompts, spec_prompt_set):  # noqa: ANN001
        assert n_prompts > 0
        assert "domain_prompt" in spec_prompt_set
        calls["ref_l"] += 1
        return ["L0"]

    def fake_build_reftext(*, cfg, n_prompts):  # noqa: ANN001
        assert n_prompts > 0
        calls["ref_t"] += 1
        return ["T0"]

    def fake_train_steering_vec_1l(**_kwargs):
        return _FakeSteeringPatch(), {"ok": True}

    def fake_eval_all_specs_glr(**_kwargs):
        return {
            "spec_metrics": {"parity_4d": {"failures": 0, "min_margin": 1.5}},
            "collateral_metrics": {"refbool_s_mean_kl": 0.01},
        }

    monkeypatch.setattr(bl, "_resolve_candidate_layers", lambda _cfg, _adapter: [0])
    monkeypatch.setattr(
        bl,
        "_glrhp_budget",
        lambda _adapter, **_kwargs: {"P_GLRHP": 10, "budget_lo": 9, "budget_hi": 11},
    )
    monkeypatch.setattr(bl, "_iter_domain_examples", fake_iter_domain_examples)
    monkeypatch.setattr(bl, "build_refbool_s", fake_build_refbool_s)
    monkeypatch.setattr(bl, "build_refbool_l", fake_build_refbool_l)
    monkeypatch.setattr(bl, "build_reftext", fake_build_reftext)
    monkeypatch.setattr(bl, "_train_steering_vec_1l", fake_train_steering_vec_1l)
    monkeypatch.setattr(bl, "_eval_all_specs_glr", fake_eval_all_specs_glr)

    results = bl.run_baselines(cfg=cfg, adapter=adapter)
    assert len(results) == 1
    assert results[0].name == "steering_vec_1l"
    assert results[0].metrics.get("skipped") is not True
    assert calls == {"ref_s": 1, "ref_l": 1, "ref_t": 1}
    runtime = cfg["_certipatch_runtime"]
    assert runtime["refbool_s_prompts"] == ["S0", "S1"]
    assert runtime["refbool_l_prompts"] == ["L0"]
    assert runtime["reftext_prompts"] == ["T0"]


def test_run_baselines_resume_from_mid_alm_solver_checkpoint(tmp_path, monkeypatch) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["baselines"] = {"enabled": ["oneshot_full_alm"]}
    adapter = _FakeAdapter()
    phase = {"name": "first"}
    seen_resume_state = {"inner_round": None}

    def fake_iter_domain_examples(_cfg, _spec_id):  # noqa: ANN001
        return [SpecExample(prompt="p0", label=1, meta={})]

    def fake_solve_constrained_minimality(*, patch, state=None, on_round_end=None, **_kwargs):  # noqa: ANN003
        if phase["name"] == "second":
            seen_resume_state["inner_round"] = None if state is None else int(state.inner_round)
        for layer in patch.cfg.candidate_layers:
            patch.params[layer]["U"] = torch.as_tensor(patch.params[layer]["U"]) + 0.01
        next_round = int((state.inner_round if state is not None else 0) + 1)
        new_state = bl.SolverState(lambda_mult=1.0, mu=1.0, inner_round=int(next_round))
        if on_round_end is not None:
            on_round_end(
                patch,
                new_state,
                {"inner_round": int(next_round - 1), "inner_round_next": int(next_round)},
            )
        if phase["name"] == "first":
            raise RuntimeError("alm-mid-crash")
        return patch, new_state, {"g_true": 0.0}

    def fake_eval_all_specs_glr(**_kwargs):  # noqa: ANN003
        return {
            "spec_metrics": {"parity_4d": {"failures": 0, "min_margin": 1.5}},
            "collateral_metrics": {"refbool_s_mean_kl": 0.01},
        }

    monkeypatch.setattr(bl, "_resolve_candidate_layers", lambda _cfg, _adapter: [0])
    monkeypatch.setattr(
        bl,
        "_glrhp_budget",
        lambda _adapter, **_kwargs: {"P_GLRHP": 10, "budget_lo": 9, "budget_hi": 11},
    )
    monkeypatch.setattr(bl, "_iter_domain_examples", fake_iter_domain_examples)
    monkeypatch.setattr(bl, "_make_glr_patch", lambda _cfg, *, cand_layers: _tiny_patch())
    monkeypatch.setattr(bl, "solve_constrained_minimality", fake_solve_constrained_minimality)
    monkeypatch.setattr(bl, "_eval_all_specs_glr", fake_eval_all_specs_glr)

    with pytest.raises(RuntimeError, match="alm-mid-crash"):
        _ = bl.run_baselines(cfg=cfg, adapter=adapter)

    ckpt_root = tmp_path / "runs" / "resume_test" / "_baselines_ckpt"
    solver_ckpt = ckpt_root / "oneshot_full_alm.solver.pt"
    assert solver_ckpt.exists()
    raw = torch.load(solver_ckpt, map_location="cpu")
    assert raw["kind"] == "baseline_alm"
    assert raw["state"]["inner_round"] == 1

    phase["name"] = "second"
    cfg["_certipatch_runtime"]["resume"] = True
    results = bl.run_baselines(cfg=cfg, adapter=adapter)
    assert len(results) == 1
    assert results[0].name == "oneshot_full_alm"
    assert seen_resume_state["inner_round"] == 1
    assert not solver_ckpt.exists()
    assert (ckpt_root / "oneshot_full_alm.json").exists()


def test_run_baselines_resume_from_mid_mo_alpha_step_checkpoint(tmp_path, monkeypatch) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["baseline"] = {"alpha_grid": [0.1]}
    adapter = _FakeAdapter()
    phase = {"name": "first"}
    seen_resume_step = {"value": None}

    def fake_iter_domain_examples(_cfg, _spec_id):  # noqa: ANN001
        return [SpecExample(prompt="p0", label=1, meta={})]

    def fake_solve_multiobjective(*, patch, alpha, resume_step=0, on_step_end=None, **_kwargs):  # noqa: ANN003
        if phase["name"] == "second":
            seen_resume_step["value"] = int(resume_step)
        if on_step_end is not None:
            on_step_end(
                patch,
                {"step_next": 37, "steps": 2000, "alpha": float(alpha), "loss": 1.23},
            )
        if phase["name"] == "first":
            raise RuntimeError("mo-mid-crash")
        return patch, {"alpha": float(alpha)}

    def fake_eval_spec_exact(**_kwargs):  # noqa: ANN003
        return SimpleNamespace(failures=0, min_margin=1.5)

    def fake_eval_collateral(**_kwargs):  # noqa: ANN003
        return SimpleNamespace(refbool_s_mean_kl=0.01)

    def fake_eval_all_specs_glr(**_kwargs):  # noqa: ANN003
        return {
            "spec_metrics": {"parity_4d": {"failures": 0, "min_margin": 1.5}},
            "collateral_metrics": {"refbool_s_mean_kl": 0.01},
        }

    monkeypatch.setattr(bl, "_resolve_candidate_layers", lambda _cfg, _adapter: [0])
    monkeypatch.setattr(
        bl,
        "_glrhp_budget",
        lambda _adapter, **_kwargs: {"P_GLRHP": 10, "budget_lo": 9, "budget_hi": 11},
    )
    monkeypatch.setattr(bl, "_iter_domain_examples", fake_iter_domain_examples)
    monkeypatch.setattr(bl, "_make_glr_patch", lambda _cfg, *, cand_layers: _tiny_patch())
    monkeypatch.setattr(bl, "solve_multiobjective", fake_solve_multiobjective)
    monkeypatch.setattr(bl, "eval_spec_exact", fake_eval_spec_exact)
    monkeypatch.setattr(bl, "eval_collateral", fake_eval_collateral)
    monkeypatch.setattr(bl, "_eval_all_specs_glr", fake_eval_all_specs_glr)

    with pytest.raises(RuntimeError, match="mo-mid-crash"):
        _ = bl.run_baselines(cfg=cfg, adapter=adapter)

    ckpt_root = tmp_path / "runs" / "resume_test" / "_baselines_ckpt"
    solver_ckpt = ckpt_root / "oneshot_full_mo" / f"alpha_{bl._alpha_tag(0.1)}.solver.pt"
    assert solver_ckpt.exists()
    raw = torch.load(solver_ckpt, map_location="cpu")
    assert raw["kind"] == "baseline_mo_alpha"
    assert raw["step_next"] == 37

    phase["name"] = "second"
    cfg["_certipatch_runtime"]["resume"] = True
    results = bl.run_baselines(cfg=cfg, adapter=adapter)
    assert len(results) == 1
    assert results[0].name == "oneshot_full_mo"
    assert seen_resume_step["value"] == 37
    assert not solver_ckpt.exists()
