"""certipatch.eval.ablations

Ablations are defined in `07_Baselines_Ablations.md` and `EXPERIMENTS.md`.

This module provides a deterministic runner used by `scripts/reproduce_paper.py` to produce
`runs/<run_id>/ablations.json` (or to embed ablation rows into paper tables).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Sequence

from certipatch.cegis.loop import cegis_result_to_jsonable, run_cegis
from certipatch.eval.metrics import eval_collateral, eval_spec_exact
from certipatch.models.load_model import ModelAdapter
from certipatch.patch_families import GLRHookPatch, GLRHPConfig
from certipatch.specs import SpecExample, SpecId


def _iter_domain_examples(cfg: Mapping[str, Any], spec_id: SpecId) -> list[SpecExample]:
    if spec_id == "compare_2d":
        from certipatch.specs.compare_2d import iter_domain

        return list(iter_domain(cfg))
    if spec_id == "parity_4d":
        from certipatch.specs.parity_4d import iter_domain

        return list(iter_domain(cfg))
    if spec_id == "balance_paren_14":
        from certipatch.specs.balance_paren_14 import iter_domain

        return list(iter_domain(cfg))
    if spec_id == "compare_6d_strat":
        from certipatch.specs.compare_6d_strat import iter_certified_coverage

        return list(iter_certified_coverage(cfg))
    raise ValueError(f"Unknown spec_id: {spec_id}")


def _make_patch(cfg: Mapping[str, Any], adapter: ModelAdapter) -> GLRHookPatch:
    patch_cfg = cfg.get("patch", {}) if isinstance(cfg.get("patch"), Mapping) else {}
    rank_r = int(patch_cfg.get("rank_r", 4))
    threshold = float(patch_cfg.get("effective_layer_threshold", 0.001))

    hook_cfg = cfg.get("hookpoints", {}) if isinstance(cfg.get("hookpoints"), Mapping) else {}
    cand_cfg = (
        hook_cfg.get("candidate_layers", {})
        if isinstance(hook_cfg.get("candidate_layers"), Mapping)
        else {}
    )
    mode = str(cand_cfg.get("mode", "quartiles"))
    explicit = cand_cfg.get("explicit")
    cand_layers = adapter.resolve_candidate_layers(
        mode, explicit=explicit if isinstance(explicit, list) else None
    )

    return GLRHookPatch(
        cfg=GLRHPConfig(
            rank_r=rank_r, candidate_layers=cand_layers, effective_layer_threshold=threshold
        )
    )


def run_ablations(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    spec_id: SpecId,
    ablations: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Run ablations for a single target spec_id and return a JSON-serializable mapping."""
    runtime = (
        cfg.get("_certipatch_runtime", {})
        if isinstance(cfg.get("_certipatch_runtime"), Mapping)
        else {}
    )
    ref_s = runtime.get("refbool_s_prompts")
    ref_l = runtime.get("refbool_l_prompts")
    ref_t = runtime.get("reftext_prompts")
    if not isinstance(ref_s, list) or not ref_s or not all(isinstance(p, str) for p in ref_s):
        raise ValueError(
            "run_ablations requires cfg['_certipatch_runtime']['refbool_s_prompts'] as a non-empty list[str]."
        )
    if not isinstance(ref_l, list) or not all(isinstance(p, str) for p in ref_l):
        ref_l = []
    if not isinstance(ref_t, list) or not all(isinstance(p, str) for p in ref_t):
        ref_t = []

    dom = _iter_domain_examples(cfg, spec_id)

    todo = (
        list(ablations)
        if ablations is not None
        else [
            "no_minimality",
            "no_cegis",
            "no_collateral",
            "no_gating",
            "rank_1",
            "single_layer",
            "random_counterexamples",
        ]
    )

    out: dict[str, Any] = {"spec_id": str(spec_id), "ablations": {}}

    for name in todo:
        ab_cfg: Dict[str, Any] = copy.deepcopy(dict(cfg))
        ab_cfg.setdefault("ablation", {})
        if not isinstance(ab_cfg["ablation"], dict):
            ab_cfg["ablation"] = {}

        # Apply ablation-specific overrides.
        if name == "no_minimality":
            ab_cfg.setdefault("cegis", {})["inner_solver"] = "multiobjective"
            ab_cfg.setdefault("multiobjective", {})["alpha"] = float(
                ab_cfg["ablation"].get("alpha", 0.1)
            )
        elif name == "no_cegis":
            ab_cfg.setdefault("cegis", {})["max_outer_iters"] = 1
            # Ensure no growth even within the single outer iteration.
            ab_cfg.setdefault("cegis", {}).setdefault("k_add", {})[str(spec_id)] = 0
        elif name == "no_collateral":
            ab_cfg["ablation"]["no_collateral"] = True
        elif name == "no_gating":
            ab_cfg.setdefault("gate", {})["force_on"] = True
        elif name == "rank_1":
            ab_cfg.setdefault("patch", {})["rank_r"] = 1
        elif name == "single_layer":
            last = int(adapter.info.n_layers) - 1
            ab_cfg.setdefault("hookpoints", {}).setdefault("candidate_layers", {})["mode"] = (
                "explicit"
            )
            ab_cfg["hookpoints"]["candidate_layers"]["explicit"] = [last]
        elif name == "random_counterexamples":
            ab_cfg.setdefault("cegis", {})["policy"] = "random"
        else:
            raise ValueError(f"Unknown ablation name: {name}")

        patch = _make_patch(ab_cfg, adapter)
        train = run_cegis(cfg=ab_cfg, adapter=adapter, spec_id=spec_id, patch=patch)
        final_patch = train.final_patch

        spec_eval = eval_spec_exact(cfg=ab_cfg, adapter=adapter, patch=final_patch, examples=dom)
        col_eval = eval_collateral(
            cfg=ab_cfg,
            adapter=adapter,
            patch=final_patch,
            refbool_s_prompts=ref_s,
            refbool_l_prompts=ref_l,
            reftext_prompts=ref_t,
        )

        out["ablations"][name] = {
            "train": cegis_result_to_jsonable(train),
            "spec_metrics": {str(spec_id): spec_eval.__dict__},
            "collateral_metrics": col_eval.__dict__,
            "patch": final_patch.serialize(),
            "config_overrides": {
                "ablation": ab_cfg.get("ablation", {}),
                "gate": ab_cfg.get("gate", {}),
                "cegis": ab_cfg.get("cegis", {}),
                "patch": ab_cfg.get("patch", {}),
                "hookpoints": ab_cfg.get("hookpoints", {}),
                "multiobjective": ab_cfg.get("multiobjective", {}),
            },
        }

    return out
