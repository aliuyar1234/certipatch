"""certipatch.cegis.loop

Counterexample-guided constraint generation (CEGIS-style) loop.

This loop is the core protocol novelty of CertiPatch:
- It is not used for sample efficiency.
- It is used to produce a compact active constraint set that supports
  constrained minimality (min collateral subject to feasibility).

The loop MUST be deterministic given config seeds.

Definitions:
- D_spec: active constraint set used by the constrained solver
- Cex: newly found counterexamples under the current patch

Stop criteria (MUST):
- Enumerable domains: stop only when failures == 0 on the full domain sweep.
- Coverage-bounded domains: stop only when failures == 0 on certified coverage sets
  and additional bounded search budgets find no counterexamples.

Fail-closed:
- If coverage is incomplete, the certificate MUST state coverage-bounded scope.
- The loop MUST not claim full-domain satisfaction for non-enumerable domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from certipatch.cegis.counterexamples import (
    Counterexample,
    find_counterexamples,
    select_hardest,
    select_random,
)
from certipatch.cegis.trainer import SolverState, solve_constrained_minimality, solve_multiobjective
from certipatch.models.load_model import ModelAdapter
from certipatch.patch_families import GLRHookPatch, GLRHPConfig
from certipatch.progress import append_jsonl, progress_enabled, run_dir, utc_now_iso
from certipatch.specs import SpecExample, SpecId


@dataclass
class CEGISResult:
    """Outputs of a CEGIS run for one spec or a union of specs."""

    final_patch: GLRHookPatch
    outer_iters: int
    cex_history: List[Dict[str, Any]]  # includes counts, hashes, minimal metadata
    active_set_size: int


def cegis_result_to_jsonable(result: CEGISResult) -> Dict[str, Any]:
    """Convert a CEGISResult into a JSON-safe payload."""
    return {
        "outer_iters": int(result.outer_iters),
        "cex_history": [dict(item) for item in result.cex_history],
        "active_set_size": int(result.active_set_size),
        "final_patch": result.final_patch.serialize(),
    }


def _stable_int_seed(*parts: str) -> int:
    h = sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return int.from_bytes(h.digest()[:8], "little", signed=False) % (2**31 - 1)


def _iter_domain_examples(cfg: Mapping[str, Any], spec_id: SpecId) -> Iterable[SpecExample]:
    if spec_id == "compare_2d":
        from certipatch.specs.compare_2d import iter_domain

        return iter_domain(cfg)
    if spec_id == "parity_4d":
        from certipatch.specs.parity_4d import iter_domain

        return iter_domain(cfg)
    if spec_id == "balance_paren_14":
        from certipatch.specs.balance_paren_14 import iter_domain

        return iter_domain(cfg)
    if spec_id == "compare_6d_strat":
        from certipatch.specs.compare_6d_strat import iter_certified_coverage

        return iter_certified_coverage(cfg)
    raise ValueError(f"Unknown spec_id: {spec_id}")


def _init_active_set(
    *,
    cfg: Mapping[str, Any],
    spec_id: SpecId,
    n0: int,
    seed: int,
) -> List[SpecExample]:
    """Initialize D_spec^(0) as a deterministic without-replacement sample in canonical order."""
    domain = list(_iter_domain_examples(cfg, spec_id))
    if not domain:
        raise ValueError(f"Empty domain for spec_id={spec_id}")
    n = int(max(1, min(int(n0), len(domain))))
    if n >= len(domain):
        # Entire domain in canonical order.
        return list(domain)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(len(domain), generator=gen).tolist()
    chosen = sorted(perm[:n])  # maintain canonical ordering
    return [domain[i] for i in chosen]


def _dedupe_examples(examples: Sequence[SpecExample]) -> List[SpecExample]:
    seen: set[str] = set()
    out: list[SpecExample] = []
    for e in examples:
        if e.prompt in seen:
            continue
        seen.add(e.prompt)
        out.append(e)
    return out


def _require_refbool_s(cfg: Mapping[str, Any]) -> Sequence[str]:
    runtime = cfg.get("_certipatch_runtime", {})
    if not isinstance(runtime, Mapping):
        runtime = {}
    prompts = runtime.get("refbool_s_prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
        raise ValueError(
            "run_cegis requires cfg['_certipatch_runtime']['refbool_s_prompts'] as a non-empty list[str]."
        )
    return prompts


def _counterexample_trace_hash(counterexamples: Sequence[Counterexample]) -> str:
    h = sha256()
    for c in counterexamples:
        h.update(c.prompt.encode("utf-8"))
        h.update(b"\t")
        h.update(str(int(c.label)).encode("ascii"))
        h.update(b"\t")
        h.update(f"{float(c.margin):.9g}".encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _with_spec_id(examples: Sequence[SpecExample], spec_id: str) -> List[SpecExample]:
    out: list[SpecExample] = []
    for e in examples:
        meta = dict(e.meta)
        meta.setdefault("spec_id", spec_id)
        out.append(SpecExample(prompt=e.prompt, label=int(e.label), meta=meta))
    return out


def _safe_tag(s: str) -> str:
    return str(s).strip().replace("/", "-").replace("\\", "-").replace(":", "-").replace(" ", "_")


def _resume_enabled(cfg: Mapping[str, Any]) -> bool:
    runtime = cfg.get("_certipatch_runtime", {})
    if not isinstance(runtime, Mapping):
        return False
    return bool(runtime.get("resume", False))


def _cegis_checkpoint_dir(cfg: Mapping[str, Any]) -> Optional[Path]:
    rd = run_dir(cfg)
    if rd is None:
        return None
    return rd / "_cegis_ckpt"


def _single_checkpoint_path(cfg: Mapping[str, Any], spec_id: SpecId) -> Optional[Path]:
    ckpt_dir = _cegis_checkpoint_dir(cfg)
    if ckpt_dir is None:
        return None
    return ckpt_dir / f"single__{_safe_tag(str(spec_id))}.pt"


def _multi_checkpoint_path(cfg: Mapping[str, Any], checkpoint_key: str) -> Optional[Path]:
    ckpt_dir = _cegis_checkpoint_dir(cfg)
    if ckpt_dir is None:
        return None
    return ckpt_dir / f"multi__{_safe_tag(checkpoint_key)}.pt"


def _state_to_payload(state: Optional[SolverState]) -> Optional[Dict[str, Any]]:
    if state is None:
        return None
    return {
        "lambda_mult": float(state.lambda_mult),
        "mu": float(state.mu),
        "inner_round": int(state.inner_round),
    }


def _state_from_payload(raw: Any) -> Optional[SolverState]:
    if not isinstance(raw, Mapping):
        return None
    try:
        return SolverState(
            lambda_mult=float(raw.get("lambda_mult", 0.0)),
            mu=float(raw.get("mu", 1.0)),
            inner_round=int(raw.get("inner_round", 0)),
        )
    except Exception:  # noqa: BLE001
        return None


def _examples_to_payload(examples: Sequence[SpecExample]) -> list[Dict[str, Any]]:
    return [
        {"prompt": str(e.prompt), "label": int(e.label), "meta": dict(e.meta)} for e in examples
    ]


def _examples_from_payload(raw: Any) -> Optional[List[SpecExample]]:
    if not isinstance(raw, list):
        return None
    out: list[SpecExample] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        prompt = item.get("prompt")
        label = item.get("label")
        meta = item.get("meta")
        if (
            not isinstance(prompt, str)
            or not isinstance(label, int)
            or not isinstance(meta, Mapping)
        ):
            return None
        out.append(
            SpecExample(
                prompt=str(prompt),
                label=int(label),
                meta={str(k): str(v) for k, v in dict(meta).items()},
            )
        )
    return out


def _patch_to_payload(patch: GLRHookPatch) -> Dict[str, Any]:
    return {
        "cfg": {
            "rank_r": int(patch.cfg.rank_r),
            "candidate_layers": [int(x) for x in patch.cfg.candidate_layers],
            "effective_layer_threshold": float(patch.cfg.effective_layer_threshold),
        },
        "params": {
            str(int(layer)): {
                "U": torch.as_tensor(vals["U"]).detach().to(device="cpu", dtype=torch.float32),
                "V": torch.as_tensor(vals["V"]).detach().to(device="cpu", dtype=torch.float32),
            }
            for layer, vals in patch.params.items()
        },
    }


def _patch_from_payload(patch: GLRHookPatch, raw: Any) -> bool:
    if not isinstance(raw, Mapping):
        return False
    cfg_raw = raw.get("cfg")
    params_raw = raw.get("params")
    if not isinstance(cfg_raw, Mapping) or not isinstance(params_raw, Mapping):
        return False
    try:
        rank_r = int(cfg_raw.get("rank_r", patch.cfg.rank_r))
        candidate_layers = [
            int(x) for x in cfg_raw.get("candidate_layers", patch.cfg.candidate_layers)
        ]
        threshold = float(
            cfg_raw.get("effective_layer_threshold", patch.cfg.effective_layer_threshold)
        )
    except Exception:  # noqa: BLE001
        return False
    patch.cfg = GLRHPConfig(
        rank_r=int(rank_r),
        candidate_layers=[int(x) for x in candidate_layers],
        effective_layer_threshold=float(threshold),
    )
    new_params: Dict[int, Dict[str, Any]] = {}
    for layer in patch.cfg.candidate_layers:
        entry = params_raw.get(str(int(layer)))
        if not isinstance(entry, Mapping):
            return False
        U = entry.get("U")
        V = entry.get("V")
        if U is None or V is None:
            return False
        new_params[int(layer)] = {
            "U": torch.as_tensor(U, dtype=torch.float32).detach().clone(),
            "V": torch.as_tensor(V, dtype=torch.float32).detach().clone(),
        }
    patch.params = new_params
    return True


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(path)


def _load_checkpoint(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        raw = torch.load(path, map_location="cpu")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, Mapping):
        return None
    return raw


def _run_cegis_multi(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    spec_ids: Sequence[SpecId],
    patch: GLRHookPatch,
    frozen_patch: Optional[GLRHookPatch] = None,
    checkpoint_key: str = "union",
) -> CEGISResult:
    """CEGIS loop over a union of specs, optionally optimizing a delta patch atop a frozen base."""
    if not spec_ids:
        raise ValueError("spec_ids must be non-empty.")

    progress_on = progress_enabled(cfg)
    run_id = ""
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), Mapping) else {}
    run_id = str(run_cfg.get("run_id", "")).strip()
    run_dir_p = run_dir(cfg) if progress_on else None
    cegis_log = (run_dir_p / "cegis_progress.jsonl") if run_dir_p is not None else None
    resume = _resume_enabled(cfg)
    ckpt_path = _multi_checkpoint_path(cfg, checkpoint_key)

    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), Mapping) else {}
    seeds = run_cfg.get("seeds", {}) if isinstance(run_cfg.get("seeds"), Mapping) else {}
    master_seed = int(seeds.get("master", 0))

    cegis_cfg = cfg.get("cegis", {}) if isinstance(cfg.get("cegis"), Mapping) else {}
    max_outer = int(cegis_cfg.get("max_outer_iters", 10))
    init_n_cfg = cegis_cfg.get("init_n", {}) if isinstance(cegis_cfg.get("init_n"), Mapping) else {}
    k_add_cfg = cegis_cfg.get("k_add", {}) if isinstance(cegis_cfg.get("k_add"), Mapping) else {}
    policy = str(cegis_cfg.get("policy", "hardest_margin"))
    inner_solver = str(cegis_cfg.get("inner_solver", "alm"))
    if inner_solver not in {"alm", "multiobjective"}:
        raise ValueError(f"Unsupported cegis inner_solver: {inner_solver}")
    if policy not in {"hardest_margin", "random"}:
        raise ValueError(f"Unsupported cegis policy: {policy}")

    D_ref = list(_require_refbool_s(cfg))

    # Initialize a per-spec sample and union them.
    D_spec: list[SpecExample] = []
    k_add_total = 0
    for sid in spec_ids:
        n0 = int(init_n_cfg.get(str(sid), 512))
        k_add_total += int(k_add_cfg.get(str(sid), 256))
        init_seed = master_seed + _stable_int_seed("cegis_init", str(sid), "union")
        D_spec.extend(
            _with_spec_id(_init_active_set(cfg=cfg, spec_id=sid, n0=n0, seed=init_seed), str(sid))
        )
    D_spec = _dedupe_examples(D_spec)

    state: Optional[SolverState] = None
    mo_step_next = 0
    history: list[dict[str, Any]] = []
    start_outer = 0
    checkpoint_done = False

    if resume and ckpt_path is not None and ckpt_path.exists():
        ckpt = _load_checkpoint(ckpt_path)
        if (
            isinstance(ckpt, Mapping)
            and str(ckpt.get("kind", "")) == "multi"
            and str(ckpt.get("checkpoint_key", "")) == str(checkpoint_key)
        ):
            loaded_patch = _patch_from_payload(patch, ckpt.get("patch"))
            loaded_examples = _examples_from_payload(ckpt.get("active_set"))
            if loaded_patch and loaded_examples is not None:
                D_spec = _dedupe_examples(loaded_examples)
                if inner_solver == "alm":
                    state = _state_from_payload(ckpt.get("state"))
                else:
                    mo_step_next = int(max(0, int(ckpt.get("mo_step_next", 0))))
                history_raw = ckpt.get("history")
                if isinstance(history_raw, list):
                    history = [dict(x) for x in history_raw if isinstance(x, Mapping)]
                start_outer = int(max(0, int(ckpt.get("outer_next", 0))))
                checkpoint_done = bool(ckpt.get("done", False))
                print(
                    f"[resume] cegis {run_id} loaded checkpoint for key={checkpoint_key} "
                    f"outer={start_outer} done={checkpoint_done}",
                    flush=True,
                )
                if cegis_log is not None:
                    append_jsonl(
                        cegis_log,
                        {
                            "timestamp_utc": utc_now_iso(),
                            "event": "cegis_resume",
                            "run_id": run_id,
                            "checkpoint_key": str(checkpoint_key),
                            "outer_next": int(start_outer),
                            "done": bool(checkpoint_done),
                            "active_set_size": int(len(D_spec)),
                            "mo_step_next": int(mo_step_next),
                        },
                    )

    outer_iters = int(start_outer)
    if cegis_log is not None:
        append_jsonl(
            cegis_log,
            {
                "timestamp_utc": utc_now_iso(),
                "event": "cegis_start",
                "run_id": run_id,
                "spec_ids": [str(s) for s in spec_ids],
                "max_outer_iters": int(max_outer),
                "policy": str(policy),
                "inner_solver": str(inner_solver),
            },
        )
        if run_id:
            print(
                f"[cegis] {run_id} start specs={','.join([str(s) for s in spec_ids])}", flush=True
            )

    if checkpoint_done or start_outer >= max_outer:
        if cegis_log is not None:
            append_jsonl(
                cegis_log,
                {
                    "timestamp_utc": utc_now_iso(),
                    "event": "cegis_end",
                    "run_id": run_id,
                    "outer_iters": int(min(start_outer, max_outer)),
                    "active_set_size": int(len(D_spec)),
                },
            )
        return CEGISResult(
            final_patch=patch,
            outer_iters=int(min(start_outer, max_outer)),
            cex_history=history,
            active_set_size=int(len(D_spec)),
        )

    for outer_iter in range(start_outer, max_outer):
        if inner_solver == "alm":
            def _on_alm_round_end(
                patch_now: GLRHookPatch,
                state_now: SolverState,
                round_meta: Mapping[str, Any],
            ) -> None:
                if ckpt_path is None:
                    return
                _write_checkpoint(
                    ckpt_path,
                    {
                        "version": 2,
                        "kind": "multi",
                        "checkpoint_key": str(checkpoint_key),
                        "spec_ids": [str(s) for s in spec_ids],
                        "outer_next": int(outer_iter),
                        "done": False,
                        "state": _state_to_payload(state_now),
                        "mo_step_next": 0,
                        "active_set": _examples_to_payload(D_spec),
                        "history": list(history),
                        "patch": _patch_to_payload(patch_now),
                        "solver_progress": {"type": "alm_round_end", **dict(round_meta)},
                    },
                )

            patch, state, diag = solve_constrained_minimality(
                cfg=cfg,
                adapter=adapter,
                patch=patch,
                frozen_patch=frozen_patch,
                D_spec=D_spec,
                D_ref=D_ref,
                state=state,
                on_round_end=_on_alm_round_end,
            )
        else:
            ab_cfg = cfg.get("ablation", {}) if isinstance(cfg.get("ablation"), Mapping) else {}
            mo_cfg = (
                cfg.get("multiobjective", {})
                if isinstance(cfg.get("multiobjective"), Mapping)
                else {}
            )
            alpha = float(mo_cfg.get("alpha", ab_cfg.get("alpha", 0.1)))
            def _on_mo_step_end(patch_now: GLRHookPatch, step_meta: Mapping[str, Any]) -> None:
                if ckpt_path is None:
                    return
                step_next = int(max(0, int(step_meta.get("step_next", 0))))
                _write_checkpoint(
                    ckpt_path,
                    {
                        "version": 2,
                        "kind": "multi",
                        "checkpoint_key": str(checkpoint_key),
                        "spec_ids": [str(s) for s in spec_ids],
                        "outer_next": int(outer_iter),
                        "done": False,
                        "state": None,
                        "mo_step_next": int(step_next),
                        "active_set": _examples_to_payload(D_spec),
                        "history": list(history),
                        "patch": _patch_to_payload(patch_now),
                        "solver_progress": {"type": "mo_step_end", **dict(step_meta)},
                    },
                )

            patch, diag = solve_multiobjective(
                cfg=cfg,
                adapter=adapter,
                patch=patch,
                frozen_patch=frozen_patch,
                D_spec=D_spec,
                D_ref=D_ref,
                alpha=alpha,
                resume_step=int(mo_step_next),
                on_step_end=_on_mo_step_end,
            )
            state = None
            mo_step_next = 0

        eval_patch = patch if frozen_patch is None else frozen_patch + patch  # type: ignore[operator]
        cex_all: list[Counterexample] = []
        per_spec: dict[str, int] = {}
        for sid in spec_ids:
            cex_sid = find_counterexamples(cfg=cfg, adapter=adapter, patch=eval_patch, spec_id=sid)
            cex_all.extend(cex_sid)
            per_spec[str(sid)] = int(len(cex_sid))

        cex_hash = _counterexample_trace_hash(cex_all)
        if policy == "hardest_margin":
            added = select_hardest(cex_all, k_add=k_add_total)
        else:
            seed = _stable_int_seed("cegis_random", "union", str(master_seed), str(outer_iter))
            added = select_random(cex_all, k_add_total, seed=seed)

        if added:
            existing_prompts = {e.prompt for e in D_spec}
            new_examples: list[SpecExample] = []
            for c in added:
                if c.prompt in existing_prompts:
                    continue
                existing_prompts.add(c.prompt)
                meta = dict(c.meta)
                meta.setdefault("spec_id", meta.get("spec_id", "unknown"))
                new_examples.append(SpecExample(prompt=c.prompt, label=int(c.label), meta=meta))
            D_spec = _dedupe_examples([*D_spec, *new_examples])

        history.append(
            {
                "outer_iter": int(outer_iter),
                "spec_ids": [str(s) for s in spec_ids],
                "active_set_size": int(len(D_spec)),
                "cex_count_total": int(len(cex_all)),
                "cex_count_per_spec": per_spec,
                "cex_hash": cex_hash,
                "added_k": int(len(added)),
                "added": [
                    {
                        "prompt": c.prompt,
                        "label": int(c.label),
                        "pred": int(c.pred),
                        "margin": float(c.margin),
                        "meta": dict(c.meta),
                    }
                    for c in added
                ],
                "diagnostics": diag,
            }
        )

        outer_iters = outer_iter + 1
        if cegis_log is not None:
            append_jsonl(
                cegis_log,
                {
                    "timestamp_utc": utc_now_iso(),
                    "event": "cegis_outer",
                    "run_id": run_id,
                    "outer_iter": int(outer_iter),
                    "active_set_size": int(len(D_spec)),
                    "cex_count_total": int(len(cex_all)),
                    "cex_count_per_spec": dict(per_spec),
                    "added_k": int(len(added)),
                    "diagnostics": diag,
                },
            )
            if run_id:
                print(
                    f"[cegis] {run_id} outer={outer_iter + 1}/{max_outer} active={len(D_spec)} "
                    f"cex={len(cex_all)} added={len(added)}",
                    flush=True,
                )

        if ckpt_path is not None:
            done_now = bool((not cex_all) or (outer_iter + 1 >= max_outer))
            _write_checkpoint(
                ckpt_path,
                {
                    "version": 2,
                    "kind": "multi",
                    "checkpoint_key": str(checkpoint_key),
                    "spec_ids": [str(s) for s in spec_ids],
                    "outer_next": int(outer_iter + 1),
                    "done": bool(done_now),
                    "state": _state_to_payload(state if inner_solver == "alm" else None),
                    "mo_step_next": 0,
                    "active_set": _examples_to_payload(D_spec),
                    "history": list(history),
                    "patch": _patch_to_payload(patch),
                },
            )
        if not cex_all:
            break

    if cegis_log is not None:
        append_jsonl(
            cegis_log,
            {
                "timestamp_utc": utc_now_iso(),
                "event": "cegis_end",
                "run_id": run_id,
                "outer_iters": int(outer_iters),
                "active_set_size": int(len(D_spec)),
            },
        )

    return CEGISResult(
        final_patch=patch,
        outer_iters=int(outer_iters),
        cex_history=history,
        active_set_size=int(len(D_spec)),
    )


def run_cegis(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    spec_id: SpecId,
    patch: GLRHookPatch,
) -> CEGISResult:
    """Run the CEGIS loop for a single spec.

    Pseudocode:
        D_spec = init_sample(spec_id, n0, seed)
        for t in range(max_outer_iters):
            patch = solve_constrained_minimality(cfg, adapter, patch, D_spec, D_ref)
            Cex = find_counterexamples(cfg, adapter, patch, spec_id)
            log history
            if Cex empty: break
             D_spec = D_spec union select_hardest(Cex, k_add)
         return result

    """
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), Mapping) else {}
    seeds = run_cfg.get("seeds", {}) if isinstance(run_cfg.get("seeds"), Mapping) else {}
    master_seed = int(seeds.get("master", 0))
    run_id = str(run_cfg.get("run_id", "")).strip()
    progress_on = progress_enabled(cfg)
    run_dir_p = run_dir(cfg) if progress_on else None
    cegis_log = (run_dir_p / "cegis_progress.jsonl") if run_dir_p is not None else None
    resume = _resume_enabled(cfg)
    ckpt_path = _single_checkpoint_path(cfg, spec_id)

    cegis_cfg = cfg.get("cegis", {}) if isinstance(cfg.get("cegis"), Mapping) else {}
    max_outer = int(cegis_cfg.get("max_outer_iters", 10))
    init_n_cfg = cegis_cfg.get("init_n", {}) if isinstance(cegis_cfg.get("init_n"), Mapping) else {}
    k_add_cfg = cegis_cfg.get("k_add", {}) if isinstance(cegis_cfg.get("k_add"), Mapping) else {}
    policy = str(cegis_cfg.get("policy", "hardest_margin"))
    inner_solver = str(cegis_cfg.get("inner_solver", "alm"))
    if inner_solver not in {"alm", "multiobjective"}:
        raise ValueError(f"Unsupported cegis inner_solver: {inner_solver}")
    if policy not in {"hardest_margin", "random"}:
        raise ValueError(f"Unsupported cegis policy: {policy}")

    n0 = int(init_n_cfg.get(str(spec_id), 512))
    k_add = int(k_add_cfg.get(str(spec_id), 256))
    if max_outer <= 0:
        raise ValueError("cfg['cegis']['max_outer_iters'] must be > 0")

    D_ref = list(_require_refbool_s(cfg))

    init_seed = master_seed + _stable_int_seed("cegis_init", str(spec_id))
    D_spec = _init_active_set(cfg=cfg, spec_id=spec_id, n0=n0, seed=init_seed)
    D_spec = _dedupe_examples(D_spec)

    state: Optional[SolverState] = None
    mo_step_next = 0
    history: list[dict[str, Any]] = []
    start_outer = 0
    checkpoint_done = False

    if resume and ckpt_path is not None and ckpt_path.exists():
        ckpt = _load_checkpoint(ckpt_path)
        if (
            isinstance(ckpt, Mapping)
            and str(ckpt.get("kind", "")) == "single"
            and str(ckpt.get("spec_id", "")) == str(spec_id)
        ):
            loaded_patch = _patch_from_payload(patch, ckpt.get("patch"))
            loaded_examples = _examples_from_payload(ckpt.get("active_set"))
            if loaded_patch and loaded_examples is not None:
                D_spec = _dedupe_examples(loaded_examples)
                if inner_solver == "alm":
                    state = _state_from_payload(ckpt.get("state"))
                else:
                    mo_step_next = int(max(0, int(ckpt.get("mo_step_next", 0))))
                history_raw = ckpt.get("history")
                if isinstance(history_raw, list):
                    history = [dict(x) for x in history_raw if isinstance(x, Mapping)]
                start_outer = int(max(0, int(ckpt.get("outer_next", 0))))
                checkpoint_done = bool(ckpt.get("done", False))
                print(
                    f"[resume] cegis {run_id} loaded checkpoint for spec={spec_id} "
                    f"outer={start_outer} done={checkpoint_done}",
                    flush=True,
                )
                if cegis_log is not None:
                    append_jsonl(
                        cegis_log,
                        {
                            "timestamp_utc": utc_now_iso(),
                            "event": "cegis_resume",
                            "run_id": run_id,
                            "spec_id": str(spec_id),
                            "outer_next": int(start_outer),
                            "done": bool(checkpoint_done),
                            "active_set_size": int(len(D_spec)),
                            "mo_step_next": int(mo_step_next),
                        },
                    )

    outer_iters = int(start_outer)
    if cegis_log is not None:
        append_jsonl(
            cegis_log,
            {
                "timestamp_utc": utc_now_iso(),
                "event": "cegis_start",
                "run_id": run_id,
                "spec_id": str(spec_id),
                "max_outer_iters": int(max_outer),
                "init_n": int(n0),
                "k_add": int(k_add),
                "policy": str(policy),
                "inner_solver": str(inner_solver),
            },
        )
        if run_id:
            print(
                f"[cegis] {run_id} spec={spec_id} start max_outer={max_outer} init_n={n0} k_add={k_add}",
                flush=True,
            )

    if checkpoint_done or start_outer >= max_outer:
        if cegis_log is not None:
            append_jsonl(
                cegis_log,
                {
                    "timestamp_utc": utc_now_iso(),
                    "event": "cegis_end",
                    "run_id": run_id,
                    "spec_id": str(spec_id),
                    "outer_iters": int(min(start_outer, max_outer)),
                    "active_set_size": int(len(D_spec)),
                },
            )
        return CEGISResult(
            final_patch=patch,
            outer_iters=int(min(start_outer, max_outer)),
            cex_history=history,
            active_set_size=int(len(D_spec)),
        )

    for outer_iter in range(start_outer, max_outer):
        if inner_solver == "alm":
            def _on_alm_round_end(
                patch_now: GLRHookPatch,
                state_now: SolverState,
                round_meta: Mapping[str, Any],
            ) -> None:
                if ckpt_path is None:
                    return
                _write_checkpoint(
                    ckpt_path,
                    {
                        "version": 2,
                        "kind": "single",
                        "spec_id": str(spec_id),
                        "outer_next": int(outer_iter),
                        "done": False,
                        "state": _state_to_payload(state_now),
                        "mo_step_next": 0,
                        "active_set": _examples_to_payload(D_spec),
                        "history": list(history),
                        "patch": _patch_to_payload(patch_now),
                        "solver_progress": {"type": "alm_round_end", **dict(round_meta)},
                    },
                )

            patch, state, diag = solve_constrained_minimality(
                cfg=cfg,
                adapter=adapter,
                patch=patch,
                D_spec=D_spec,
                D_ref=D_ref,
                state=state,
                on_round_end=_on_alm_round_end,
            )
        else:
            ab_cfg = cfg.get("ablation", {}) if isinstance(cfg.get("ablation"), Mapping) else {}
            mo_cfg = (
                cfg.get("multiobjective", {})
                if isinstance(cfg.get("multiobjective"), Mapping)
                else {}
            )
            alpha = float(mo_cfg.get("alpha", ab_cfg.get("alpha", 0.1)))
            def _on_mo_step_end(patch_now: GLRHookPatch, step_meta: Mapping[str, Any]) -> None:
                if ckpt_path is None:
                    return
                step_next = int(max(0, int(step_meta.get("step_next", 0))))
                _write_checkpoint(
                    ckpt_path,
                    {
                        "version": 2,
                        "kind": "single",
                        "spec_id": str(spec_id),
                        "outer_next": int(outer_iter),
                        "done": False,
                        "state": None,
                        "mo_step_next": int(step_next),
                        "active_set": _examples_to_payload(D_spec),
                        "history": list(history),
                        "patch": _patch_to_payload(patch_now),
                        "solver_progress": {"type": "mo_step_end", **dict(step_meta)},
                    },
                )

            patch, diag = solve_multiobjective(
                cfg=cfg,
                adapter=adapter,
                patch=patch,
                frozen_patch=None,
                D_spec=D_spec,
                D_ref=D_ref,
                alpha=alpha,
                resume_step=int(mo_step_next),
                on_step_end=_on_mo_step_end,
            )
            state = None
            mo_step_next = 0

        cex = find_counterexamples(cfg=cfg, adapter=adapter, patch=patch, spec_id=spec_id)
        cex_hash = _counterexample_trace_hash(cex)
        failures_count = int(sum(1 for c in cex if int(c.pred) != int(c.label)))

        added: list[Counterexample] = []
        if cex:
            if policy == "hardest_margin":
                added = select_hardest(cex, k_add=k_add)
            else:
                seed = _stable_int_seed(
                    "cegis_random", str(spec_id), str(master_seed), str(outer_iter)
                )
                added = select_random(cex, k_add, seed=seed)

            # Add to active set as labeled constraints, de-duplicated by prompt string.
            existing_prompts = {e.prompt for e in D_spec}
            new_examples: list[SpecExample] = []
            for c in added:
                if c.prompt in existing_prompts:
                    continue
                existing_prompts.add(c.prompt)
                meta = dict(c.meta)
                meta.setdefault("spec_id", str(spec_id))
                new_examples.append(SpecExample(prompt=c.prompt, label=int(c.label), meta=meta))
            D_spec = _dedupe_examples([*D_spec, *new_examples])

        history.append(
            {
                "outer_iter": int(outer_iter),
                "spec_id": str(spec_id),
                "active_set_size": int(len(D_spec)),
                "cex_count": int(len(cex)),
                "failures_count": int(failures_count),
                "cex_hash": cex_hash,
                "added_k": int(len(added)),
                "added": [
                    {
                        "prompt": c.prompt,
                        "label": int(c.label),
                        "pred": int(c.pred),
                        "margin": float(c.margin),
                        "meta": dict(c.meta),
                    }
                    for c in added
                ],
                "diagnostics": diag,
            }
        )

        outer_iters = outer_iter + 1
        if cegis_log is not None:
            append_jsonl(
                cegis_log,
                {
                    "timestamp_utc": utc_now_iso(),
                    "event": "cegis_outer",
                    "run_id": run_id,
                    "spec_id": str(spec_id),
                    "outer_iter": int(outer_iter),
                    "active_set_size": int(len(D_spec)),
                    "cex_count": int(len(cex)),
                    "failures_count": int(failures_count),
                    "added_k": int(len(added)),
                    "cex_hash": str(cex_hash),
                    "diagnostics": diag,
                },
            )
            if run_id:
                g_true = diag.get("g_true") if isinstance(diag, Mapping) else None
                print(
                    f"[cegis] {run_id} spec={spec_id} outer={outer_iter + 1}/{max_outer} "
                    f"active={len(D_spec)} cex={len(cex)} added={len(added)} "
                    f"fail={failures_count} g_true={g_true}",
                    flush=True,
                )

        if ckpt_path is not None:
            done_now = bool((not cex) or (outer_iter + 1 >= max_outer))
            _write_checkpoint(
                ckpt_path,
                {
                    "version": 2,
                    "kind": "single",
                    "spec_id": str(spec_id),
                    "outer_next": int(outer_iter + 1),
                    "done": bool(done_now),
                    "state": _state_to_payload(state if inner_solver == "alm" else None),
                    "mo_step_next": 0,
                    "active_set": _examples_to_payload(D_spec),
                    "history": list(history),
                    "patch": _patch_to_payload(patch),
                },
            )
        if not cex:
            break

    if cegis_log is not None:
        append_jsonl(
            cegis_log,
            {
                "timestamp_utc": utc_now_iso(),
                "event": "cegis_end",
                "run_id": run_id,
                "spec_id": str(spec_id),
                "outer_iters": int(outer_iters),
                "active_set_size": int(len(D_spec)),
            },
        )

    return CEGISResult(
        final_patch=patch,
        outer_iters=int(outer_iters),
        cex_history=history,
        active_set_size=int(len(D_spec)),
    )


def run_compositionality_suite(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch_factory: Any,
) -> Dict[str, Any]:
    """Run the compositionality experiment (A-only, B-only, A+B, A→B, B→A, Joint AB).

    Required specs:
      - Spec A = compare_2d
      - Spec B = parity_4d

    Output MUST include:
      - For each condition: failure counts for A and B, collateral metrics, patch complexity metrics.

    """
    from certipatch.eval.metrics import eval_collateral, eval_spec_exact

    comp_cfg = (
        cfg.get("compositionality", {}) if isinstance(cfg.get("compositionality"), Mapping) else {}
    )
    spec_A = comp_cfg.get("spec_A", "compare_2d")
    spec_B = comp_cfg.get("spec_B", "parity_4d")
    if spec_A not in {"compare_2d", "parity_4d"} or spec_B not in {"compare_2d", "parity_4d"}:
        raise ValueError(
            "compositionality suite requires spec_A/spec_B in {'compare_2d','parity_4d'}."
        )
    spec_A_id = spec_A  # type: ignore[assignment]
    spec_B_id = spec_B  # type: ignore[assignment]

    conditions = comp_cfg.get(
        "conditions",
        ["A_only", "B_only", "A_plus_B", "A_then_B", "B_then_A", "Joint_AB"],
    )
    if not isinstance(conditions, list) or not all(isinstance(c, str) for c in conditions):
        raise ValueError("cfg['compositionality']['conditions'] must be a list[str].")

    D_ref_s = cfg.get("_certipatch_runtime", {}).get("refbool_s_prompts", [])  # type: ignore[assignment]
    D_ref_l = cfg.get("_certipatch_runtime", {}).get("refbool_l_prompts", [])  # type: ignore[assignment]
    D_ref_t = cfg.get("_certipatch_runtime", {}).get("reftext_prompts", [])  # type: ignore[assignment]

    # Learn base patches for A and B.
    results: dict[str, CEGISResult] = {}
    if {"A_only", "A_plus_B", "A_then_B", "Joint_AB"} & set(conditions):
        results["A"] = run_cegis(cfg=cfg, adapter=adapter, spec_id=spec_A_id, patch=patch_factory())
    if {"B_only", "A_plus_B", "B_then_A", "Joint_AB"} & set(conditions):
        results["B"] = run_cegis(cfg=cfg, adapter=adapter, spec_id=spec_B_id, patch=patch_factory())
    patch_A = results["A"].final_patch if "A" in results else None
    patch_B = results["B"].final_patch if "B" in results else None

    def _eval_condition(p: GLRHookPatch) -> Dict[str, Any]:
        A_dom = list(_iter_domain_examples(cfg, spec_A_id))
        B_dom = list(_iter_domain_examples(cfg, spec_B_id))
        A_m = eval_spec_exact(cfg=cfg, adapter=adapter, patch=p, examples=A_dom)
        B_m = eval_spec_exact(cfg=cfg, adapter=adapter, patch=p, examples=B_dom)
        col = eval_collateral(
            cfg=cfg,
            adapter=adapter,
            patch=p,
            refbool_s_prompts=D_ref_s if isinstance(D_ref_s, list) else [],
            refbool_l_prompts=D_ref_l if isinstance(D_ref_l, list) else [],
            reftext_prompts=D_ref_t if isinstance(D_ref_t, list) else [],
        )
        return {
            "spec_A": A_m.__dict__,
            "spec_B": B_m.__dict__,
            "collateral": col.__dict__,
            "patch": p.serialize(),
        }

    out: dict[str, Any] = {"spec_A": spec_A_id, "spec_B": spec_B_id, "conditions": {}}

    for cond in conditions:
        if cond == "A_only":
            if patch_A is None:
                raise ValueError("Missing patch_A for A_only.")
            out["conditions"][cond] = {
                "train": cegis_result_to_jsonable(results["A"]),
                **_eval_condition(patch_A),
            }
            continue
        if cond == "B_only":
            if patch_B is None:
                raise ValueError("Missing patch_B for B_only.")
            out["conditions"][cond] = {
                "train": cegis_result_to_jsonable(results["B"]),
                **_eval_condition(patch_B),
            }
            continue
        if cond == "A_plus_B":
            if patch_A is None or patch_B is None:
                raise ValueError("Missing patch_A/patch_B for A_plus_B.")
            out["conditions"][cond] = {
                "train": {
                    "A": cegis_result_to_jsonable(results["A"]),
                    "B": cegis_result_to_jsonable(results["B"]),
                },
                **_eval_condition(patch_A + patch_B),  # type: ignore[operator]
            }
            continue

        if cond == "A_then_B":
            if patch_A is None:
                raise ValueError("Missing patch_A for A_then_B.")
            delta_res = _run_cegis_multi(
                cfg=cfg,
                adapter=adapter,
                spec_ids=[spec_A_id, spec_B_id],
                patch=patch_factory(),
                frozen_patch=patch_A,
                checkpoint_key="A_then_B",
            )
            out["conditions"][cond] = {
                "train": {
                    "base": cegis_result_to_jsonable(results["A"]),
                    "delta": cegis_result_to_jsonable(delta_res),
                },
                **_eval_condition(patch_A + delta_res.final_patch),  # type: ignore[operator]
            }
            continue
        if cond == "B_then_A":
            if patch_B is None:
                raise ValueError("Missing patch_B for B_then_A.")
            delta_res = _run_cegis_multi(
                cfg=cfg,
                adapter=adapter,
                spec_ids=[spec_A_id, spec_B_id],
                patch=patch_factory(),
                frozen_patch=patch_B,
                checkpoint_key="B_then_A",
            )
            out["conditions"][cond] = {
                "train": {
                    "base": cegis_result_to_jsonable(results["B"]),
                    "delta": cegis_result_to_jsonable(delta_res),
                },
                **_eval_condition(patch_B + delta_res.final_patch),  # type: ignore[operator]
            }
            continue
        if cond == "Joint_AB":
            joint = _run_cegis_multi(
                cfg=cfg,
                adapter=adapter,
                spec_ids=[spec_A_id, spec_B_id],
                patch=patch_factory(),
                frozen_patch=None,
                checkpoint_key="Joint_AB",
            )
            out["conditions"][cond] = {
                "train": cegis_result_to_jsonable(joint),
                **_eval_condition(joint.final_patch),
            }
            continue
        raise ValueError(f"Unknown compositionality condition: {cond}")

    return out
