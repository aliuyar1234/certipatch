"""certipatch.cegis.trainer

Constrained minimality solver: minimize collateral subject to zero spec violations.

Objective:
    minimize   L_col(phi) + R(phi)
    subject to g(phi; D_spec) = 0

Where g is a margin-based max violation function over the active set D_spec.

This module specifies the augmented Lagrangian method (ALM) schedule used in CertiPatch.

ALM form:
    L_AL(phi; lambda, mu) =
        L_col(phi) + R(phi) + lambda*g(phi) + (mu/2)*g(phi)^2

Schedule (fixed by config):
- mu_init
- mu_mult_on_violation
- mu_div_on_feasible
- mu_floor
- inner_steps_per_outer
- max_inner_rounds

Determinism:
- Every optimization step MUST be deterministic given seeds and device settings.
- Use deterministic PyTorch operations where possible; record flags in run_record.

Fail-closed:
- If g cannot reach 0 on an enumerable domain and a baseline can, report the failure
  explicitly; do not silently change thresholds.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch

from certipatch.hooks import (
    GateSpec,
    answer_positions,
    apply_hookpoint_patch,
    boolqa_gate,
    gather_logits_at_positions,
)
from certipatch.models.load_model import ModelAdapter, assert_or_select_answer_tokens
from certipatch.patch_families import GLRHookPatch
from certipatch.progress import (
    append_jsonl,
    progress_config,
    progress_enabled,
    run_dir,
    utc_now_iso,
)
from certipatch.specs import SpecExample


@dataclass
class SolverState:
    lambda_mult: float
    mu: float
    inner_round: int


def _select_from_grid(
    cfg: Mapping[str, Any], *, scalar_key: str, grid_key: str, default: float
) -> float:
    if scalar_key in cfg:
        return float(cfg[scalar_key])
    grid = cfg.get(grid_key)
    if isinstance(grid, list) and grid:
        return float(grid[len(grid) // 2])
    return float(default)


def _sanitize_scalar(
    value: float,
    *,
    fallback: float,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> float:
    out = float(value)
    if not math.isfinite(out):
        out = float(fallback)
    if lo is not None:
        out = max(float(lo), out)
    if hi is not None:
        out = min(float(hi), out)
    return float(out)


def _kl_pq_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(p||q) where p=softmax(logits_p), q=softmax(logits_q) computed in float32."""
    logp = torch.log_softmax(logits_p.float(), dim=-1)
    logq = torch.log_softmax(logits_q.float(), dim=-1)
    p = logp.exp()
    return (p * (logp - logq)).sum(dim=-1)


def _regularizer(patch: GLRHookPatch, *, lambda_l2: float, lambda_group: float) -> torch.Tensor:
    if not patch.params:
        return torch.tensor(0.0)

    total_l2 = torch.tensor(0.0, device=next(iter(patch.params.values()))["U"].device)
    total_group = torch.tensor(0.0, device=total_l2.device)

    for layer in patch.cfg.candidate_layers:
        if layer not in patch.params:
            continue
        U = patch.params[layer]["U"]
        V = patch.params[layer]["V"]
        norm2 = (U.pow(2).sum() + V.pow(2).sum()).float()
        total_l2 = total_l2 + norm2
        total_group = total_group + torch.sqrt(norm2 + 1e-12)

    return float(lambda_l2) * total_l2 + float(lambda_group) * total_group


def _make_patch_delta_fn(
    patch: GLRHookPatch, frozen_patch: Optional[GLRHookPatch], *, layer: int
) -> Callable[[Any], Any]:
    def _fn(h: Any) -> Any:
        h_t = torch.as_tensor(h)
        out = h_t + patch.delta_vectors(h_t, layer=layer)
        if frozen_patch is not None:
            out = out + frozen_patch.delta_vectors(h_t, layer=layer).detach()
        return out

    return _fn


def _g_smooth_from_v(v: torch.Tensor, *, beta: float, formula: str) -> torch.Tensor:
    """Smooth approximation to max(v_i) for non-negative v_i.

    Configurable via `objective.g_smooth_formula`:
      - "log_mean_exp": (logsumexp(beta*v) - log(n))/beta  (batch-size invariant)
      - "logsumexp":    logsumexp(beta*v)/beta
    """
    if v.numel() == 0:
        raise ValueError("Empty violation tensor (unexpected).")

    if formula == "log_mean_exp":
        return (torch.logsumexp(float(beta) * v, dim=0) - math.log(int(v.numel()))) / float(beta)
    if formula == "logsumexp":
        return torch.logsumexp(float(beta) * v, dim=0) / float(beta)
    raise ValueError(
        f"Unknown objective.g_smooth_formula: {formula!r} (expected 'log_mean_exp' or 'logsumexp')."
    )


def _compute_margins(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch: GLRHookPatch,
    frozen_patch: Optional[GLRHookPatch],
    examples: Sequence[SpecExample],
    yes_id: int,
    no_id: int,
    gate: GateSpec,
    kind: str,
    cand_layers: Sequence[int],
) -> torch.Tensor:
    """Return margins for a batch of spec examples on the patched model (no_grad-safe)."""
    prompts = [e.prompt for e in examples]
    labels = torch.tensor([int(e.label) for e in examples], dtype=torch.int64)

    gate_pred = [boolqa_gate(p, gate) for p in prompts]
    if not all(gate_pred):
        bad = gate_pred.index(False)
        raise ValueError(f"Spec example out-of-scope (gate=false) at index {bad}.")

    toks = adapter.tokenize(prompts)
    input_ids = toks["input_ids"]
    attention_mask = toks["attention_mask"]
    p_idx = answer_positions(attention_mask)

    gate_cfg = cfg.get("gate", {}) if isinstance(cfg.get("gate"), Mapping) else {}
    gate_enabled = bool(gate_cfg.get("enabled", True))
    gate_force_on = bool(gate_cfg.get("force_on", False))
    gate_mask = torch.tensor(gate_pred, dtype=torch.bool, device=p_idx.device)
    if gate_force_on:
        gate_mask = torch.ones_like(gate_mask)
    elif not gate_enabled:
        gate_mask = torch.zeros_like(gate_mask)

    handles = []
    try:
        if patch.params:
            for layer in cand_layers:
                handles.append(
                    apply_hookpoint_patch(
                        adapter,
                        kind=kind,
                        layer=int(layer),
                        batch_positions=p_idx,
                        gate_mask=gate_mask,
                        patch_fn=_make_patch_delta_fn(patch, frozen_patch, layer=int(layer)),
                    )
                )
        logits = adapter.forward_logits(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            try:
                h.handle.remove()
            except Exception:  # noqa: BLE001
                pass

    logits_p = gather_logits_at_positions(logits, p_idx)
    yes_logits = logits_p[:, yes_id]
    no_logits = logits_p[:, no_id]
    labels_dev = labels.to(device=yes_logits.device)
    correct_logits = torch.where(labels_dev == 1, yes_logits, no_logits)
    incorrect_logits = torch.where(labels_dev == 1, no_logits, yes_logits)
    return (correct_logits - incorrect_logits).float()


def _eval_g_true(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch: GLRHookPatch,
    frozen_patch: Optional[GLRHookPatch],
    D_spec: Sequence[SpecExample],
    yes_id: int,
    no_id: int,
    gate: GateSpec,
    kind: str,
    cand_layers: Sequence[int],
    tau: float,
    batch_size: int,
) -> Dict[str, Any]:
    tau_t = float(tau)
    violations = 0
    g_true = 0.0
    min_margin = float("inf")

    all_margins: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(D_spec), batch_size):
            end = min(len(D_spec), start + batch_size)
            margins = _compute_margins(
                cfg=cfg,
                adapter=adapter,
                patch=patch,
                frozen_patch=frozen_patch,
                examples=D_spec[start:end],
                yes_id=yes_id,
                no_id=no_id,
                gate=gate,
                kind=kind,
                cand_layers=cand_layers,
            )
            all_margins.append(margins.detach().cpu())
            min_margin = min(min_margin, float(margins.min().item()))
            v = torch.relu(torch.tensor(tau_t) - margins)
            violations += int((v > 0).sum().item())
            g_true = max(g_true, float(v.max().item()))

    margins_full = torch.cat(all_margins, dim=0).numpy()
    p05_margin = float(torch.quantile(torch.from_numpy(margins_full), 0.05).item())

    return {
        "g_true": float(g_true),
        "violations": int(violations),
        "min_margin": float(min_margin),
        "p05_margin": float(p05_margin),
    }


def _eval_ref_kl(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch: GLRHookPatch,
    frozen_patch: Optional[GLRHookPatch],
    D_ref: Sequence[str],
    gate: GateSpec,
    kind: str,
    cand_layers: Sequence[int],
    batch_size: int,
    max_eval: int,
) -> float:
    if not D_ref:
        return 0.0

    gate_pred = [boolqa_gate(p, gate) for p in D_ref[:max_eval]]
    if not all(gate_pred):
        bad = gate_pred.index(False)
        raise ValueError(f"Reference prompt out-of-scope (gate=false) at index {bad}.")

    gate_cfg = cfg.get("gate", {}) if isinstance(cfg.get("gate"), Mapping) else {}
    gate_enabled = bool(gate_cfg.get("enabled", True))
    gate_force_on = bool(gate_cfg.get("force_on", False))
    total_kl = 0.0
    total_n = 0

    with torch.no_grad():
        for start in range(0, min(len(D_ref), max_eval), batch_size):
            end = min(min(len(D_ref), max_eval), start + batch_size)
            prompts = list(D_ref[start:end])
            toks = adapter.tokenize(prompts)
            input_ids = toks["input_ids"]
            attention_mask = toks["attention_mask"]
            p_idx = answer_positions(attention_mask)

            with torch.no_grad():
                logits_base = adapter.forward_logits(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            logits_base_p = gather_logits_at_positions(logits_base, p_idx)

            gate_mask = torch.tensor(gate_pred[start:end], dtype=torch.bool, device=p_idx.device)
            if gate_force_on:
                gate_mask = torch.ones_like(gate_mask)
            elif not gate_enabled:
                gate_mask = torch.zeros_like(gate_mask)

            handles = []
            try:
                if patch.params:
                    for layer in cand_layers:
                        handles.append(
                            apply_hookpoint_patch(
                                adapter,
                                kind=kind,
                                layer=int(layer),
                                batch_positions=p_idx,
                                gate_mask=gate_mask,
                                patch_fn=_make_patch_delta_fn(
                                    patch, frozen_patch, layer=int(layer)
                                ),
                            )
                        )
                logits_patch = adapter.forward_logits(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            finally:
                for h in handles:
                    try:
                        h.handle.remove()
                    except Exception:  # noqa: BLE001
                        pass

            logits_patch_p = gather_logits_at_positions(logits_patch, p_idx)
            kl = _kl_pq_from_logits(logits_base_p, logits_patch_p).float().mean()

            total_kl += float(kl.item()) * (end - start)
            total_n += end - start

    return total_kl / max(1, total_n)


def solve_constrained_minimality(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch: GLRHookPatch,
    frozen_patch: Optional[GLRHookPatch] = None,
    D_spec: Sequence[SpecExample],
    D_ref: Sequence[str],
    state: Optional[SolverState] = None,
    on_round_end: Optional[Callable[[GLRHookPatch, SolverState, Mapping[str, Any]], None]] = None,
) -> Tuple[GLRHookPatch, SolverState, Dict[str, Any]]:
    """Solve the constrained minimality problem for the current active set.

    Inputs:
      - D_spec: active constraints (labeled prompts)
      - D_ref: reference prompts for collateral (unlabeled; gate must fire)

    Output:
      - updated patch
      - updated solver state
      - diagnostics dict: g value, L_col, norm, effective layers, etc.

    Pseudocode sketch:
        if state is None: lambda=0, mu=mu_init
        repeat for max_inner_rounds:
            for step in inner_steps_per_outer:
                sample minibatch from D_spec and D_ref deterministically
                compute g_batch (approx of max violation) with smoothing beta
                compute L_col_batch (KL)
                compute R
                compute L_AL and take Adam step
            compute g_full on D_spec
            update lambda <- lambda + mu*g_full
            if g_full > 0: mu <- max(mu_floor, mu * mu_mult_on_violation)
            else: mu <- max(mu_floor, mu / mu_div_on_feasible)
            if g_full == 0 and collateral stops improving: break

    Implementation notes:
      - Use a smooth approximation to max over D_spec for gradients, controlled by beta.
      - Enforce g==0 exactly using full-batch evaluation of D_spec at the end of each round.
      - Never declare feasibility on minibatches only.

    """
    if not D_spec:
        raise ValueError("D_spec must be non-empty.")
    if not D_ref:
        raise ValueError("D_ref must be non-empty.")

    progress_on = progress_enabled(cfg)
    progress_cfg = progress_config(cfg)
    log_every_steps = max(1, int(progress_cfg.get("log_every_steps", 100)))
    run_id = ""
    run_cfg = cfg.get("run", {})
    if isinstance(run_cfg, Mapping):
        run_id = str(run_cfg.get("run_id", "")).strip()
    run_dir_p = run_dir(cfg) if progress_on else None
    train_log = (run_dir_p / "train_progress.jsonl") if run_dir_p is not None else None

    ablation_cfg = cfg.get("ablation", {}) if isinstance(cfg.get("ablation"), Mapping) else {}
    disable_collateral = bool(ablation_cfg.get("no_collateral", False))

    gate_cfg = cfg.get("gate", {})
    if not isinstance(gate_cfg, Mapping):
        raise ValueError("cfg['gate'] must be a mapping.")
    gate = GateSpec(wrapper_line=str(gate_cfg["wrapper_line"]), suffix=str(gate_cfg["suffix"]))

    tokens = assert_or_select_answer_tokens(adapter, cfg)
    yes_id = int(tokens["yes_id"])
    no_id = int(tokens["no_id"])

    hook_cfg = cfg.get("hookpoints", {})
    if not isinstance(hook_cfg, Mapping):
        raise ValueError("cfg['hookpoints'] must be a mapping.")
    kind = str(hook_cfg.get("kind", "resid_post"))

    cand_cfg = hook_cfg.get("candidate_layers", {})
    if not isinstance(cand_cfg, Mapping):
        raise ValueError("cfg['hookpoints']['candidate_layers'] must be a mapping.")
    mode = str(cand_cfg.get("mode", "quartiles"))
    explicit = cand_cfg.get("explicit")
    resolved_layers = adapter.resolve_candidate_layers(
        mode, explicit=explicit if isinstance(explicit, list) else None
    )
    if patch.cfg.candidate_layers and sorted(patch.cfg.candidate_layers) != sorted(resolved_layers):
        raise ValueError(
            "Patch candidate layers do not match cfg['hookpoints']['candidate_layers'] resolution."
        )

    cand_layers = patch.cfg.candidate_layers or resolved_layers

    seed = 0
    if isinstance(run_cfg, Mapping):
        seeds = run_cfg.get("seeds", {})
        if isinstance(seeds, Mapping):
            seed = int(seeds.get("torch", seeds.get("master", 0)))

    if not patch.params:
        patch.init_parameters(d_model=adapter.info.d_model, seed=seed)

    if frozen_patch is not None:
        if not frozen_patch.params:
            raise ValueError("frozen_patch must have initialized parameters.")
        if sorted(frozen_patch.cfg.candidate_layers) != sorted(cand_layers):
            raise ValueError(
                "frozen_patch candidate_layers must match trainable patch candidate_layers."
            )

    # Make parameters trainable leaf tensors on the model device.
    device = adapter.tokenize([D_ref[0]])["input_ids"].device
    dtype = torch.float32
    for layer in cand_layers:
        if layer not in patch.params:
            raise ValueError(f"Patch missing parameters for layer {layer}")
        for k in ("U", "V"):
            t = patch.params[layer][k]
            patch.params[layer][k] = torch.nn.Parameter(
                torch.as_tensor(t).detach().to(device=device, dtype=dtype)
            )

    # Hyperparameters
    objective_cfg = cfg.get("objective", {}) if isinstance(cfg.get("objective"), Mapping) else {}
    tau = float(objective_cfg.get("tau_margin", 1.0))
    beta = float(objective_cfg.get("beta_smooth", 50.0))
    g_smooth_formula = str(objective_cfg.get("g_smooth_formula", "log_mean_exp")).strip()
    if g_smooth_formula not in ("log_mean_exp", "logsumexp"):
        raise ValueError(
            f"cfg['objective']['g_smooth_formula'] must be 'log_mean_exp' or 'logsumexp', got {g_smooth_formula!r}"
        )

    regularizer_cfg = (
        cfg.get("regularizers", {}) if isinstance(cfg.get("regularizers"), Mapping) else {}
    )
    lambda_l2 = _select_from_grid(
        regularizer_cfg, scalar_key="lambda_l2", grid_key="lambda_l2_grid", default=1e-4
    )
    lambda_group = _select_from_grid(
        regularizer_cfg, scalar_key="lambda_group", grid_key="lambda_group_grid", default=1e-3
    )

    optimizer_cfg = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer"), Mapping) else {}
    lr = _select_from_grid(optimizer_cfg, scalar_key="lr", grid_key="lr_grid", default=3e-3)
    betas = optimizer_cfg.get("adam_betas", [0.9, 0.999])
    if not (isinstance(betas, list) and len(betas) == 2):
        raise ValueError("cfg['optimizer']['adam_betas'] must be a list of two floats.")
    adam_betas = (float(betas[0]), float(betas[1]))
    inner_steps = int(optimizer_cfg.get("inner_steps_per_outer", 2000))
    max_inner_rounds = int(optimizer_cfg.get("max_inner_rounds", 5))
    patience_steps = int(optimizer_cfg.get("patience_steps", 200))
    grad_clip_norm = float(optimizer_cfg.get("grad_clip_norm", 10.0))
    if not math.isfinite(grad_clip_norm) or grad_clip_norm < 0.0:
        raise ValueError("cfg['optimizer']['grad_clip_norm'] must be a finite non-negative float.")

    alm_cfg = cfg.get("alm", {}) if isinstance(cfg.get("alm"), Mapping) else {}
    mu_init = float(alm_cfg.get("mu_init", 1.0))
    mu_mult_on_violation = float(alm_cfg.get("mu_mult_on_violation", 10.0))
    mu_div_on_feasible = float(alm_cfg.get("mu_div_on_feasible", 2.0))
    mu_floor = float(alm_cfg.get("mu_floor", 1e-3))
    mu_ceiling = float(alm_cfg.get("mu_ceiling", 1e12))
    lambda_ceiling = float(alm_cfg.get("lambda_ceiling", 1e12))
    non_finite_backoff_factor = float(alm_cfg.get("non_finite_backoff_factor", 10.0))
    non_finite_max_backoffs = int(alm_cfg.get("non_finite_max_backoffs", 8))
    if not math.isfinite(mu_init) or mu_init <= 0.0:
        raise ValueError("cfg['alm']['mu_init'] must be a finite positive float.")
    if not math.isfinite(mu_mult_on_violation) or mu_mult_on_violation <= 0.0:
        raise ValueError("cfg['alm']['mu_mult_on_violation'] must be a finite positive float.")
    if not math.isfinite(mu_div_on_feasible) or mu_div_on_feasible <= 0.0:
        raise ValueError("cfg['alm']['mu_div_on_feasible'] must be a finite positive float.")
    if not math.isfinite(mu_floor) or mu_floor < 0.0:
        raise ValueError("cfg['alm']['mu_floor'] must be a finite non-negative float.")
    if not math.isfinite(mu_ceiling) or mu_ceiling <= 0.0:
        raise ValueError("cfg['alm']['mu_ceiling'] must be a finite positive float.")
    if mu_ceiling < mu_floor:
        raise ValueError("cfg['alm']['mu_ceiling'] must be >= cfg['alm']['mu_floor'].")
    if not math.isfinite(lambda_ceiling) or lambda_ceiling <= 0.0:
        raise ValueError("cfg['alm']['lambda_ceiling'] must be a finite positive float.")
    if not math.isfinite(non_finite_backoff_factor) or non_finite_backoff_factor <= 1.0:
        raise ValueError("cfg['alm']['non_finite_backoff_factor'] must be > 1.0.")
    if non_finite_max_backoffs < 0:
        raise ValueError("cfg['alm']['non_finite_max_backoffs'] must be >= 0.")

    eval_batch_size = (
        int(cfg.get("evaluation", {}).get("batch_size", 64))
        if isinstance(cfg.get("evaluation"), Mapping)
        else 64
    )
    spec_batch_size = max(1, int(optimizer_cfg.get("spec_batch_size", eval_batch_size)))
    ref_batch_size = max(1, int(optimizer_cfg.get("ref_batch_size", eval_batch_size)))

    # ALM state
    if state is None:
        lambda_mult = 0.0
        mu = mu_init
        round_offset = 0
    else:
        lambda_mult = float(state.lambda_mult)
        mu = float(state.mu)
        # NOTE: inner_round is tracked only for determinism (minibatch permutation seeding) and reporting.
        # It MUST NOT be used to skip optimization across CEGIS outer iterations.
        round_offset = int(state.inner_round)
    lambda_mult = _sanitize_scalar(
        lambda_mult, fallback=0.0, lo=-float(lambda_ceiling), hi=float(lambda_ceiling)
    )
    mu = _sanitize_scalar(mu, fallback=mu_init, lo=mu_floor, hi=mu_ceiling)

    # Pre-validate gate for ref prompts (fail-closed).
    ref_gate_pred = [boolqa_gate(p, gate) for p in D_ref]
    if not all(ref_gate_pred):
        bad = ref_gate_pred.index(False)
        raise ValueError(f"D_ref prompt out-of-scope (gate=false) at index {bad}.")

    # Deterministic minibatch permutations.
    spec_idx = list(range(len(D_spec)))
    ref_idx = list(range(len(D_ref)))

    best_ref_kl = float("inf")
    steps_since_improve = 0
    last_step_kl = float("nan")
    last_step_reg = float("nan")
    last_step_g = float("nan")
    last_step_loss = float("nan")
    last_step_grad_norm = float("nan")
    non_finite_events = 0

    g_stats: Dict[str, Any] = {}
    inner_rounds_used = 0
    t0 = time.monotonic()

    for inner_round in range(0, max_inner_rounds):
        global_round = round_offset + inner_round
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 10_000 * global_round)
        spec_perm = torch.randperm(len(spec_idx), generator=gen).tolist()
        ref_perm = torch.randperm(len(ref_idx), generator=gen).tolist()
        spec_pos = 0
        ref_pos = 0

        params = []
        for layer in cand_layers:
            params.append(patch.params[layer]["U"])
            params.append(patch.params[layer]["V"])
        optimizer = torch.optim.Adam(params, lr=lr, betas=adam_betas)

        for step in range(inner_steps):
            # Batch selection (deterministic, cyclic).
            spec_batch_ids = [
                spec_perm[(spec_pos + i) % len(spec_perm)] for i in range(spec_batch_size)
            ]
            spec_pos = (spec_pos + spec_batch_size) % len(spec_perm)
            ref_batch_ids = [ref_perm[(ref_pos + i) % len(ref_perm)] for i in range(ref_batch_size)]
            ref_pos = (ref_pos + ref_batch_size) % len(ref_perm)

            spec_batch = [D_spec[i] for i in spec_batch_ids]
            ref_batch = [D_ref[i] for i in ref_batch_ids]

            optimizer.zero_grad(set_to_none=True)

            # Collateral: KL(p_base || p_patched) at answer position.
            if disable_collateral:
                kl = torch.tensor(0.0, device=device, dtype=torch.float32)
            else:
                toks_ref = adapter.tokenize(ref_batch)
                ref_input_ids = toks_ref["input_ids"]
                ref_attention_mask = toks_ref["attention_mask"]
                p_ref = answer_positions(ref_attention_mask)

                gate_mask_ref = torch.tensor(
                    [True] * len(ref_batch), dtype=torch.bool, device=p_ref.device
                )
                if bool(gate_cfg.get("force_on", False)):
                    gate_mask_ref = torch.ones_like(gate_mask_ref)
                elif not bool(gate_cfg.get("enabled", True)):
                    gate_mask_ref = torch.zeros_like(gate_mask_ref)

                with torch.no_grad():
                    logits_base = adapter.forward_logits(
                        input_ids=ref_input_ids, attention_mask=ref_attention_mask
                    )
                logits_base_p = gather_logits_at_positions(logits_base, p_ref)

                handles = []
                try:
                    for layer in cand_layers:
                        handles.append(
                            apply_hookpoint_patch(
                                adapter,
                                kind=kind,
                                layer=int(layer),
                                batch_positions=p_ref,
                                gate_mask=gate_mask_ref,
                                patch_fn=_make_patch_delta_fn(
                                    patch, frozen_patch, layer=int(layer)
                                ),
                            )
                        )
                    logits_patch = adapter.forward_logits(
                        input_ids=ref_input_ids, attention_mask=ref_attention_mask
                    )
                finally:
                    for h in handles:
                        try:
                            h.handle.remove()
                        except Exception:  # noqa: BLE001
                            pass

                logits_patch_p = gather_logits_at_positions(logits_patch, p_ref)
                kl = _kl_pq_from_logits(logits_base_p, logits_patch_p).mean()

            # Constraint batch: smooth max violation proxy.
            margins = _compute_margins(
                cfg=cfg,
                adapter=adapter,
                patch=patch,
                frozen_patch=frozen_patch,
                examples=spec_batch,
                yes_id=yes_id,
                no_id=no_id,
                gate=gate,
                kind=kind,
                cand_layers=cand_layers,
            )
            v = torch.relu(torch.tensor(tau, device=margins.device) - margins)
            if v.numel() == 0:
                raise ValueError("Empty spec batch (unexpected).")

            g_smooth = _g_smooth_from_v(v, beta=beta, formula=g_smooth_formula)

            reg = _regularizer(patch, lambda_l2=lambda_l2, lambda_group=lambda_group).to(
                device=margins.device
            )
            lambda_mult = _sanitize_scalar(
                lambda_mult, fallback=0.0, lo=-float(lambda_ceiling), hi=float(lambda_ceiling)
            )
            mu = _sanitize_scalar(mu, fallback=mu_init, lo=mu_floor, hi=mu_ceiling)
            loss = (
                kl + reg + float(lambda_mult) * g_smooth + (float(mu) / 2.0) * g_smooth * g_smooth
            )

            if not torch.isfinite(loss):
                non_finite_events += 1
                if non_finite_events > non_finite_max_backoffs:
                    raise ValueError(
                        "Non-finite loss encountered during ALM optimization after "
                        f"{non_finite_events} backoff attempts."
                    )
                lambda_mult = _sanitize_scalar(
                    float(lambda_mult) / non_finite_backoff_factor,
                    fallback=0.0,
                    lo=-float(lambda_ceiling),
                    hi=float(lambda_ceiling),
                )
                mu = _sanitize_scalar(
                    float(mu) / non_finite_backoff_factor,
                    fallback=mu_init,
                    lo=mu_floor,
                    hi=mu_ceiling,
                )
                if train_log is not None:
                    append_jsonl(
                        train_log,
                        {
                            "timestamp_utc": utc_now_iso(),
                            "event": "alm_non_finite_backoff",
                            "run_id": run_id,
                            "inner_round": int(inner_round),
                            "step": int(step + 1),
                            "loss": float(loss.detach().cpu().item()),
                            "g_smooth": float(g_smooth.detach().cpu().item()),
                            "kl": float(kl.detach().cpu().item()),
                            "reg": float(reg.detach().cpu().item()),
                            "lambda_mult_next": float(lambda_mult),
                            "mu_next": float(mu),
                            "attempt": int(non_finite_events),
                        },
                    )
                if run_id:
                    print(
                        f"[alm] {run_id} non-finite loss at r{inner_round + 1} step {step + 1}; "
                        f"backoff attempt {non_finite_events}/{non_finite_max_backoffs} "
                        f"mu->{mu:.3g} lam->{lambda_mult:.3g}",
                        flush=True,
                    )
                continue

            loss.backward()
            grad_norm = float("nan")
            if grad_clip_norm > 0.0:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)
                grad_norm = float(torch.as_tensor(grad_norm_t).detach().cpu().item())
                if not math.isfinite(grad_norm):
                    non_finite_events += 1
                    if non_finite_events > non_finite_max_backoffs:
                        raise ValueError(
                            "Non-finite gradient norm encountered during ALM optimization after "
                            f"{non_finite_events} backoff attempts."
                        )
                    optimizer.zero_grad(set_to_none=True)
                    lambda_mult = _sanitize_scalar(
                        float(lambda_mult) / non_finite_backoff_factor,
                        fallback=0.0,
                        lo=-float(lambda_ceiling),
                        hi=float(lambda_ceiling),
                    )
                    mu = _sanitize_scalar(
                        float(mu) / non_finite_backoff_factor,
                        fallback=mu_init,
                        lo=mu_floor,
                        hi=mu_ceiling,
                    )
                    if train_log is not None:
                        append_jsonl(
                            train_log,
                            {
                                "timestamp_utc": utc_now_iso(),
                                "event": "alm_non_finite_grad_backoff",
                                "run_id": run_id,
                                "inner_round": int(inner_round),
                                "step": int(step + 1),
                                "grad_norm": float(grad_norm),
                                "lambda_mult_next": float(lambda_mult),
                                "mu_next": float(mu),
                                "attempt": int(non_finite_events),
                            },
                        )
                    if run_id:
                        print(
                            f"[alm] {run_id} non-finite grad at r{inner_round + 1} step {step + 1}; "
                            f"backoff attempt {non_finite_events}/{non_finite_max_backoffs} "
                            f"mu->{mu:.3g} lam->{lambda_mult:.3g}",
                            flush=True,
                        )
                    continue
            optimizer.step()

            last_step_kl = float(kl.detach().cpu().item())
            last_step_reg = float(reg.detach().cpu().item())
            last_step_g = float(g_smooth.detach().cpu().item())
            last_step_loss = float(loss.detach().cpu().item())
            last_step_grad_norm = float(grad_norm)

            if train_log is not None and (
                step == 0 or (step + 1) % log_every_steps == 0 or step + 1 == inner_steps
            ):
                cuda_mem = None
                cuda_max_mem = None
                if device.type == "cuda" and torch.cuda.is_available():
                    try:
                        cuda_mem = int(torch.cuda.memory_allocated(device))
                        cuda_max_mem = int(torch.cuda.max_memory_allocated(device))
                    except Exception:  # noqa: BLE001
                        cuda_mem = None
                        cuda_max_mem = None
                append_jsonl(
                    train_log,
                    {
                        "timestamp_utc": utc_now_iso(),
                        "event": "alm_step",
                        "run_id": run_id,
                        "inner_round": int(inner_round),
                        "step": int(step + 1),
                        "inner_steps": int(inner_steps),
                        "elapsed_s": float(time.monotonic() - t0),
                        "loss": float(last_step_loss),
                        "g_smooth": float(last_step_g),
                        "g_smooth_formula": str(g_smooth_formula),
                        "kl": float(last_step_kl),
                        "reg": float(last_step_reg),
                        "lambda_mult": float(lambda_mult),
                        "mu": float(mu),
                        "grad_norm": float(last_step_grad_norm),
                        "cuda_mem_bytes": cuda_mem,
                        "cuda_max_mem_bytes": cuda_max_mem,
                    },
                )
                if run_id:
                    print(
                        f"[alm] {run_id} r{inner_round + 1}/{max_inner_rounds} "
                        f"step {step + 1}/{inner_steps} "
                        f"loss={last_step_loss:.3g} g={last_step_g:.3g} kl={last_step_kl:.3g} "
                        f"mu={mu:.3g} lam={lambda_mult:.3g} grad={last_step_grad_norm:.3g}",
                        flush=True,
                    )

            # Optional plateau tracking on a fixed subset of D_ref.
            if not disable_collateral and patience_steps > 0 and (step + 1) % patience_steps == 0:
                ref_eval = _eval_ref_kl(
                    cfg=cfg,
                    adapter=adapter,
                    patch=patch,
                    frozen_patch=frozen_patch,
                    D_ref=D_ref,
                    gate=gate,
                    kind=kind,
                    cand_layers=cand_layers,
                    batch_size=ref_batch_size,
                    max_eval=min(len(D_ref), 512),
                )
                if ref_eval + 1e-5 < best_ref_kl:
                    best_ref_kl = ref_eval
                    steps_since_improve = 0
                else:
                    steps_since_improve += patience_steps

        g_stats = _eval_g_true(
            cfg=cfg,
            adapter=adapter,
            patch=patch,
            frozen_patch=frozen_patch,
            D_spec=D_spec,
            yes_id=yes_id,
            no_id=no_id,
            gate=gate,
            kind=kind,
            cand_layers=cand_layers,
            tau=tau,
            batch_size=spec_batch_size,
        )
        g_true = float(g_stats["g_true"])
        inner_rounds_used = inner_round + 1

        # Update ALM multipliers.
        lambda_mult = _sanitize_scalar(
            float(lambda_mult + mu * g_true),
            fallback=0.0,
            lo=-float(lambda_ceiling),
            hi=float(lambda_ceiling),
        )
        if g_true > 0:
            mu = _sanitize_scalar(
                float(mu * mu_mult_on_violation),
                fallback=mu_init,
                lo=mu_floor,
                hi=mu_ceiling,
            )
        else:
            mu = _sanitize_scalar(
                float(mu / mu_div_on_feasible),
                fallback=mu_init,
                lo=mu_floor,
                hi=mu_ceiling,
            )

        if train_log is not None:
            append_jsonl(
                train_log,
                {
                    "timestamp_utc": utc_now_iso(),
                    "event": "alm_round_end",
                    "run_id": run_id,
                    "inner_round": int(inner_round),
                    "elapsed_s": float(time.monotonic() - t0),
                    "g_true": float(g_true),
                    "violations": int(g_stats.get("violations", 0)),
                    "min_margin": float(g_stats.get("min_margin", float("nan"))),
                    "p05_margin": float(g_stats.get("p05_margin", float("nan"))),
                    "lambda_mult": float(lambda_mult),
                    "mu": float(mu),
                    "best_ref_kl": float(best_ref_kl),
                    "steps_since_improve": int(steps_since_improve),
                    "non_finite_events": int(non_finite_events),
                },
            )
            if run_id:
                print(
                    f"[alm] {run_id} r{inner_round + 1}/{max_inner_rounds} end "
                    f"g_true={g_true:.3g} viol={int(g_stats.get('violations', 0))} "
                    f"mu={mu:.3g} lam={lambda_mult:.3g}",
                    flush=True,
                )

        round_state = SolverState(
            lambda_mult=float(lambda_mult),
            mu=float(mu),
            inner_round=int(round_offset + inner_round + 1),
        )
        if on_round_end is not None:
            on_round_end(
                patch,
                round_state,
                {
                    "inner_round": int(inner_round),
                    "inner_round_next": int(round_state.inner_round),
                    "elapsed_s": float(time.monotonic() - t0),
                    "g_true": float(g_true),
                    "violations": int(g_stats.get("violations", 0)),
                    "min_margin": float(g_stats.get("min_margin", float("nan"))),
                    "p05_margin": float(g_stats.get("p05_margin", float("nan"))),
                    "lambda_mult": float(lambda_mult),
                    "mu": float(mu),
                    "best_ref_kl": float(best_ref_kl),
                    "steps_since_improve": int(steps_since_improve),
                    "non_finite_events": int(non_finite_events),
                },
            )

        if g_true == 0.0 and steps_since_improve >= patience_steps:
            break

    g_true_out: float | None = None
    violations_out: int | None = None
    min_margin_out: float | None = None
    p05_margin_out: float | None = None
    if g_stats:
        g_true_raw = g_stats.get("g_true")
        violations_raw = g_stats.get("violations")
        min_margin_raw = g_stats.get("min_margin")
        p05_margin_raw = g_stats.get("p05_margin")
        if (
            g_true_raw is None
            or violations_raw is None
            or min_margin_raw is None
            or p05_margin_raw is None
        ):
            raise ValueError("Missing g_stats fields (g_true/violations/min_margin/p05_margin).")
        g_true_out = float(g_true_raw)
        violations_out = int(violations_raw)
        min_margin_out = float(min_margin_raw)
        p05_margin_out = float(p05_margin_raw)

    diagnostics: Dict[str, Any] = {
        "g_true": g_true_out,
        "violations": violations_out,
        "min_margin": min_margin_out,
        "p05_margin": p05_margin_out,
        "kl_last": last_step_kl,
        "reg_last": last_step_reg,
        "g_smooth_last": last_step_g,
        "loss_last": last_step_loss,
        "grad_norm_last": last_step_grad_norm,
        "lambda_mult": float(lambda_mult),
        "mu": float(mu),
        "inner_rounds": int(inner_rounds_used),
        "non_finite_events": int(non_finite_events),
        "hyperparams": {
            "lr": float(lr),
            "lambda_l2": float(lambda_l2),
            "lambda_group": float(lambda_group),
            "tau_margin": float(tau),
            "beta_smooth": float(beta),
            "g_smooth_formula": str(g_smooth_formula),
            "grad_clip_norm": float(grad_clip_norm),
            "mu_ceiling": float(mu_ceiling),
            "lambda_ceiling": float(lambda_ceiling),
            "non_finite_backoff_factor": float(non_finite_backoff_factor),
            "non_finite_max_backoffs": int(non_finite_max_backoffs),
        },
        "patch": patch.serialize(),
    }

    new_state = SolverState(
        lambda_mult=float(lambda_mult),
        mu=float(mu),
        inner_round=int(round_offset + inner_rounds_used),
    )
    return patch, new_state, diagnostics


def solve_multiobjective(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch: GLRHookPatch,
    frozen_patch: Optional[GLRHookPatch] = None,
    D_spec: Sequence[SpecExample],
    D_ref: Sequence[str],
    alpha: float,
    resume_step: int = 0,
    on_step_end: Optional[Callable[[GLRHookPatch, Mapping[str, Any]], None]] = None,
) -> Tuple[GLRHookPatch, Dict[str, Any]]:
    """One-shot multiobjective optimizer (no ALM, no CEGIS growth).

    Loss:
        L = g_smooth(D_spec) + alpha * L_col(D_ref) + R(patch)

    This is used for:
      - OneShot-FullDomain-MO baseline
      - NoMinimality-style ablations (when run inside a larger protocol)
    """
    if not D_spec:
        raise ValueError("D_spec must be non-empty.")
    if float(alpha) != 0.0 and not D_ref:
        raise ValueError("D_ref must be non-empty when alpha != 0.")

    progress_on = progress_enabled(cfg)
    progress_cfg = progress_config(cfg)
    log_every_steps = max(1, int(progress_cfg.get("log_every_steps", 100)))
    run_id = ""
    run_cfg = cfg.get("run", {})
    if isinstance(run_cfg, Mapping):
        run_id = str(run_cfg.get("run_id", "")).strip()
    run_dir_p = run_dir(cfg) if progress_on else None
    train_log = (run_dir_p / "train_progress.jsonl") if run_dir_p is not None else None
    t0 = time.monotonic()

    gate_cfg = cfg.get("gate", {})
    if not isinstance(gate_cfg, Mapping):
        raise ValueError("cfg['gate'] must be a mapping.")
    gate = GateSpec(wrapper_line=str(gate_cfg["wrapper_line"]), suffix=str(gate_cfg["suffix"]))

    tokens = assert_or_select_answer_tokens(adapter, cfg)
    yes_id = int(tokens["yes_id"])
    no_id = int(tokens["no_id"])

    hook_cfg = cfg.get("hookpoints", {})
    if not isinstance(hook_cfg, Mapping):
        raise ValueError("cfg['hookpoints'] must be a mapping.")
    kind = str(hook_cfg.get("kind", "resid_post"))

    cand_cfg = hook_cfg.get("candidate_layers", {})
    if not isinstance(cand_cfg, Mapping):
        raise ValueError("cfg['hookpoints']['candidate_layers'] must be a mapping.")
    mode = str(cand_cfg.get("mode", "quartiles"))
    explicit = cand_cfg.get("explicit")
    cand_layers = adapter.resolve_candidate_layers(
        mode, explicit=explicit if isinstance(explicit, list) else None
    )

    if not patch.params:
        seed_cfg = (
            cfg.get("run", {}).get("seeds", {}) if isinstance(cfg.get("run"), Mapping) else {}
        )
        seed = int(seed_cfg.get("torch", seed_cfg.get("master", 0)))
        patch.init_parameters(d_model=adapter.info.d_model, seed=seed)

    if frozen_patch is not None:
        if not frozen_patch.params:
            raise ValueError("frozen_patch must have initialized parameters.")
        if sorted(frozen_patch.cfg.candidate_layers) != sorted(cand_layers):
            raise ValueError(
                "frozen_patch candidate_layers must match trainable patch candidate_layers."
            )

    # Make parameters trainable leaf tensors on the model device.
    device = adapter.tokenize([D_spec[0].prompt])["input_ids"].device
    dtype = torch.float32
    for layer in cand_layers:
        if layer not in patch.params:
            raise ValueError(f"Patch missing parameters for layer {layer}")
        for k in ("U", "V"):
            t = patch.params[layer][k]
            patch.params[layer][k] = torch.nn.Parameter(
                torch.as_tensor(t).detach().to(device=device, dtype=dtype)
            )

    objective_cfg = cfg.get("objective", {}) if isinstance(cfg.get("objective"), Mapping) else {}
    tau = float(objective_cfg.get("tau_margin", 1.0))
    beta = float(objective_cfg.get("beta_smooth", 50.0))
    g_smooth_formula = str(objective_cfg.get("g_smooth_formula", "log_mean_exp")).strip()
    if g_smooth_formula not in ("log_mean_exp", "logsumexp"):
        raise ValueError(
            f"cfg['objective']['g_smooth_formula'] must be 'log_mean_exp' or 'logsumexp', got {g_smooth_formula!r}"
        )

    regularizer_cfg = (
        cfg.get("regularizers", {}) if isinstance(cfg.get("regularizers"), Mapping) else {}
    )
    lambda_l2 = _select_from_grid(
        regularizer_cfg, scalar_key="lambda_l2", grid_key="lambda_l2_grid", default=1e-4
    )
    lambda_group = _select_from_grid(
        regularizer_cfg, scalar_key="lambda_group", grid_key="lambda_group_grid", default=1e-3
    )

    optimizer_cfg = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer"), Mapping) else {}
    lr = _select_from_grid(optimizer_cfg, scalar_key="lr", grid_key="lr_grid", default=3e-3)
    betas = optimizer_cfg.get("adam_betas", [0.9, 0.999])
    if not (isinstance(betas, list) and len(betas) == 2):
        raise ValueError("cfg['optimizer']['adam_betas'] must be a list of two floats.")
    adam_betas = (float(betas[0]), float(betas[1]))
    steps = int(optimizer_cfg.get("inner_steps_per_outer", 2000))
    steps = max(1, steps)
    start_step = int(max(0, min(int(resume_step), steps)))

    eval_batch_size = (
        int(cfg.get("evaluation", {}).get("batch_size", 64))
        if isinstance(cfg.get("evaluation"), Mapping)
        else 64
    )
    spec_batch_size = max(1, int(optimizer_cfg.get("spec_batch_size", eval_batch_size)))
    ref_batch_size = max(1, int(optimizer_cfg.get("ref_batch_size", eval_batch_size)))

    params = []
    for layer in cand_layers:
        params.append(patch.params[layer]["U"])
        params.append(patch.params[layer]["V"])
    optimizer = torch.optim.Adam(params, lr=lr, betas=adam_betas)

    # Deterministic minibatch permutations.
    seed_cfg = cfg.get("run", {}).get("seeds", {}) if isinstance(cfg.get("run"), Mapping) else {}
    master_seed = int(seed_cfg.get("master", 0))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(master_seed + 12345)
    spec_perm = torch.randperm(len(D_spec), generator=gen).tolist()
    ref_perm = torch.randperm(len(D_ref), generator=gen).tolist() if D_ref else [0]
    spec_pos = (start_step * spec_batch_size) % len(spec_perm)
    ref_pos = (start_step * ref_batch_size) % len(ref_perm) if D_ref else 0

    last_kl = 0.0
    last_g = 0.0
    last_reg = 0.0

    if start_step > 0 and run_id:
        print(f"[resume] mo {run_id} step {start_step}/{steps}", flush=True)

    for _step in range(start_step, steps):
        spec_ids = [spec_perm[(spec_pos + i) % len(spec_perm)] for i in range(spec_batch_size)]
        spec_pos = (spec_pos + spec_batch_size) % len(spec_perm)
        ref_ids = (
            [ref_perm[(ref_pos + i) % len(ref_perm)] for i in range(ref_batch_size)]
            if D_ref
            else []
        )
        ref_pos = (ref_pos + ref_batch_size) % len(ref_perm)

        spec_batch = [D_spec[i] for i in spec_ids]
        ref_batch = [D_ref[i] for i in ref_ids] if ref_ids else []

        optimizer.zero_grad(set_to_none=True)

        if float(alpha) != 0.0:
            toks_ref = adapter.tokenize(ref_batch)
            ref_input_ids = toks_ref["input_ids"]
            ref_attention_mask = toks_ref["attention_mask"]
            p_ref = answer_positions(ref_attention_mask)

            gate_mask_ref = torch.tensor(
                [True] * len(ref_batch), dtype=torch.bool, device=p_ref.device
            )
            if bool(gate_cfg.get("force_on", False)):
                gate_mask_ref = torch.ones_like(gate_mask_ref)
            elif not bool(gate_cfg.get("enabled", True)):
                gate_mask_ref = torch.zeros_like(gate_mask_ref)

            with torch.no_grad():
                logits_base = adapter.forward_logits(
                    input_ids=ref_input_ids, attention_mask=ref_attention_mask
                )
            logits_base_p = gather_logits_at_positions(logits_base, p_ref)

            handles = []
            try:
                for layer in cand_layers:
                    handles.append(
                        apply_hookpoint_patch(
                            adapter,
                            kind=kind,
                            layer=int(layer),
                            batch_positions=p_ref,
                            gate_mask=gate_mask_ref,
                            patch_fn=_make_patch_delta_fn(patch, frozen_patch, layer=int(layer)),
                        )
                    )
                logits_patch = adapter.forward_logits(
                    input_ids=ref_input_ids, attention_mask=ref_attention_mask
                )
            finally:
                for h in handles:
                    try:
                        h.handle.remove()
                    except Exception:  # noqa: BLE001
                        pass

            logits_patch_p = gather_logits_at_positions(logits_patch, p_ref)
            kl = _kl_pq_from_logits(logits_base_p, logits_patch_p).mean()
        else:
            kl = torch.tensor(0.0, device=device, dtype=torch.float32)

        margins = _compute_margins(
            cfg=cfg,
            adapter=adapter,
            patch=patch,
            frozen_patch=frozen_patch,
            examples=spec_batch,
            yes_id=yes_id,
            no_id=no_id,
            gate=gate,
            kind=kind,
            cand_layers=cand_layers,
        )
        v = torch.relu(torch.tensor(tau, device=margins.device) - margins)
        g_smooth = _g_smooth_from_v(v, beta=beta, formula=g_smooth_formula)

        reg = _regularizer(patch, lambda_l2=lambda_l2, lambda_group=lambda_group).to(
            device=margins.device
        )

        loss = g_smooth + float(alpha) * kl + reg
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss encountered during multiobjective optimization.")

        loss.backward()
        optimizer.step()

        last_kl = float(kl.detach().cpu().item())
        last_g = float(g_smooth.detach().cpu().item())
        last_reg = float(reg.detach().cpu().item())

        if train_log is not None and (
            _step == 0 or (_step + 1) % log_every_steps == 0 or _step + 1 == steps
        ):
            append_jsonl(
                train_log,
                {
                    "timestamp_utc": utc_now_iso(),
                    "event": "mo_step",
                    "run_id": run_id,
                    "step": int(_step + 1),
                    "steps": int(steps),
                    "elapsed_s": float(time.monotonic() - t0),
                    "loss": float(loss.detach().cpu().item()),
                    "g_smooth": float(last_g),
                    "g_smooth_formula": str(g_smooth_formula),
                    "kl": float(last_kl),
                    "reg": float(last_reg),
                    "alpha": float(alpha),
                },
            )
            if run_id:
                print(
                    f"[mo] {run_id} step {_step + 1}/{steps} loss={float(loss.detach().cpu().item()):.3g} "
                    f"g={last_g:.3g} kl={last_kl:.3g}",
                    flush=True,
                )

        if on_step_end is not None and (
            _step == start_step or (_step + 1) % log_every_steps == 0 or _step + 1 == steps
        ):
            on_step_end(
                patch,
                {
                    "step_next": int(_step + 1),
                    "steps": int(steps),
                    "alpha": float(alpha),
                    "elapsed_s": float(time.monotonic() - t0),
                    "loss": float(loss.detach().cpu().item()),
                    "g_smooth": float(last_g),
                    "kl": float(last_kl),
                    "reg": float(last_reg),
                },
            )

    g_stats = _eval_g_true(
        cfg=cfg,
        adapter=adapter,
        patch=patch,
        frozen_patch=frozen_patch,
        D_spec=D_spec,
        yes_id=yes_id,
        no_id=no_id,
        gate=gate,
        kind=kind,
        cand_layers=cand_layers,
        tau=tau,
        batch_size=spec_batch_size,
    )

    diagnostics: Dict[str, Any] = {
        "alpha": float(alpha),
        "g_true": float(g_stats.get("g_true", 0.0)),
        "violations": int(g_stats.get("violations", 0)),
        "min_margin": float(g_stats.get("min_margin", 0.0)),
        "p05_margin": float(g_stats.get("p05_margin", 0.0)),
        "kl_last": float(last_kl),
        "g_smooth_last": float(last_g),
        "g_smooth_formula": str(g_smooth_formula),
        "reg_last": float(last_reg),
        "steps": int(steps),
        "resume_step": int(start_step),
        "steps_run": int(max(0, steps - start_step)),
        "lr": float(lr),
        "lambda_l2": float(lambda_l2),
        "lambda_group": float(lambda_group),
        "tau_margin": float(tau),
        "beta_smooth": float(beta),
    }
    return patch, diagnostics
