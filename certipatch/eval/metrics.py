"""certipatch.eval.metrics

Metric definitions are part of SSOT. Implementations MUST match these definitions.

Spec metrics:
- failures: count of prompts where predicted label != Spec(label) under patched model
- pass_rate: 1 - failures/total
- margins: defined over answer token logits:
      margin = logit(correct) - logit(incorrect)

Collateral metrics (gate-firing reference suites):
- RefBool-S: mean KL divergence at answer position:
      KL(p_base(.|x) || p_patched(.|x))
  computed at the answer position p(x).
- RefBool-L: long-horizon drift:
      - divergence_rate: fraction of prompts where generated text differs
      - first_diff_index: mean index of first differing token
      - norm_edit_distance: normalized edit distance on strings

Generation semantics for RefBool-L (MUST):
- Apply patch only on the prompt forward pass at the answer position.
- Then generate greedily with cache enabled and patch disabled in subsequent decode steps.
  This measures how a repaired prompt representation propagates, without repeatedly injecting
  deltas at every step.

Answer position p(x) MUST be computed per-example:
    p = attention_mask.sum(dim=1) - 1

Fail-closed:
- If any suite prompt is out-of-scope (gate false), exclude it from the suite by construction.
- If the suite generator produces an out-of-scope prompt, abort and fix generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
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
from certipatch.specs import SpecExample


@dataclass(frozen=True)
class SpecMetrics:
    total: int
    failures: int
    pass_rate: float
    min_margin: float
    p05_margin: float


@dataclass(frozen=True)
class StratumMetrics:
    total: int
    failures: int
    pass_rate: float


@dataclass(frozen=True)
class CollateralMetrics:
    refbool_s_mean_kl: float
    refbool_s_ci95: Tuple[float, float]
    refbool_l_divergence_rate: float
    refbool_l_first_diff_index: float
    refbool_l_norm_edit_distance: float
    reftext_mean_kl: float


def _kl_pq_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(p||q) where p=softmax(logits_p), q=softmax(logits_q), computed in float32."""
    logp = torch.log_softmax(logits_p.float(), dim=-1)
    logq = torch.log_softmax(logits_q.float(), dim=-1)
    p = logp.exp()
    return (p * (logp - logq)).sum(dim=-1)


def _bootstrap_ci95(values: np.ndarray, *, resamples: int, seed: int) -> Tuple[float, float]:
    n = int(values.shape[0])
    if n <= 0:
        return (0.0, 0.0)
    r = int(max(1, resamples))
    rng = np.random.default_rng(int(seed))
    means = np.empty((r,), dtype=np.float64)
    for i in range(r):
        idx = rng.integers(0, n, size=(n,), dtype=np.int64)
        means[i] = float(values[idx].mean())
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def _token_edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    # Levenshtein distance on token IDs; O(len(a)*len(b)) with a small memory footprint.
    if a == b:
        return 0
    n = len(a)
    m = len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def _first_diff_index(a: Sequence[int], b: Sequence[int], *, default: int) -> int:
    k = min(len(a), len(b))
    for i in range(k):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return k
    return int(default)


def _resolve_candidate_layers_for_patch(
    cfg: Mapping[str, Any], adapter: ModelAdapter, patch: GLRHookPatch
) -> list[int]:
    if not patch.params:
        return []
    hook_cfg = cfg.get("hookpoints", {})
    if not isinstance(hook_cfg, Mapping):
        raise ValueError("cfg['hookpoints'] must be a mapping.")
    cand_cfg = hook_cfg.get("candidate_layers", {})
    if not isinstance(cand_cfg, Mapping):
        raise ValueError(
            "cfg['hookpoints']['candidate_layers'] must be a mapping when patch is applied."
        )
    mode = str(cand_cfg.get("mode", "quartiles"))
    explicit = cand_cfg.get("explicit")
    cand_layers = adapter.resolve_candidate_layers(
        mode, explicit=explicit if isinstance(explicit, list) else None
    )
    missing = [layer for layer in cand_layers if layer not in patch.params]
    if missing:
        raise ValueError(f"Patch missing parameters for candidate layers: {missing}")
    return cand_layers


def _make_patch_fn(patch: GLRHookPatch, *, layer: int) -> Callable[[Any], Any]:
    def _fn(h: Any) -> Any:
        return patch.apply_to_vectors(h, layer=layer)

    return _fn


def _mean_kl_at_answer_position(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch: GLRHookPatch,
    prompts: Sequence[str],
    require_gate: bool,
    kind: str,
    cand_layers: Sequence[int],
    batch_size: int,
) -> Tuple[float, np.ndarray]:
    if not prompts:
        return (0.0, np.zeros((0,), dtype=np.float64))

    gate_cfg = cfg.get("gate", {})
    if not isinstance(gate_cfg, Mapping):
        raise ValueError("cfg['gate'] must be a mapping.")
    gate = GateSpec(wrapper_line=str(gate_cfg["wrapper_line"]), suffix=str(gate_cfg["suffix"]))
    gate_enabled = bool(gate_cfg.get("enabled", True))
    gate_force_on = bool(gate_cfg.get("force_on", False))

    gate_pred_all = [boolqa_gate(p, gate) for p in prompts]
    if require_gate and not all(gate_pred_all):
        bad = gate_pred_all.index(False)
        raise ValueError(f"Suite prompt at index {bad} is out-of-scope (gate=false).")
    if not require_gate and any(gate_pred_all):
        bad = gate_pred_all.index(True)
        raise ValueError(f"RefText prompt at index {bad} unexpectedly in-scope (gate=true).")

    kls: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            end = min(len(prompts), start + batch_size)
            batch = list(prompts[start:end])
            toks = adapter.tokenize(batch)
            input_ids = toks["input_ids"]
            attention_mask = toks["attention_mask"]
            p_idx = answer_positions(attention_mask)

            logits_base = adapter.forward_logits(input_ids=input_ids, attention_mask=attention_mask)
            logits_base_p = gather_logits_at_positions(logits_base, p_idx)

            gate_mask = torch.tensor(
                gate_pred_all[start:end], dtype=torch.bool, device=p_idx.device
            )
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
                                patch_fn=_make_patch_fn(patch, layer=int(layer)),
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
            kl = (
                _kl_pq_from_logits(logits_base_p, logits_patch_p)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            kls.append(kl)

    kl_all = np.concatenate(kls, axis=0) if kls else np.zeros((0,), dtype=np.float64)
    return (float(kl_all.mean()) if kl_all.size else 0.0, kl_all)


def _generate_greedy_continuation(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch: Optional[GLRHookPatch],
    prompt: str,
    kind: str,
    cand_layers: Sequence[int],
    fixed_prompt_pos: int,
) -> list[int]:
    gen_cfg = (
        cfg.get("evaluation", {}).get("generation", {})
        if isinstance(cfg.get("evaluation"), Mapping)
        else {}
    )
    if not isinstance(gen_cfg, Mapping):
        gen_cfg = {}
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 128))
    max_new_tokens = max(0, max_new_tokens)

    eos_id = getattr(getattr(adapter, "tokenizer", None), "eos_token_id", None)
    eos_id = int(eos_id) if eos_id is not None else None

    gate_cfg = cfg.get("gate", {})
    if not isinstance(gate_cfg, Mapping):
        raise ValueError("cfg['gate'] must be a mapping.")
    gate = GateSpec(wrapper_line=str(gate_cfg["wrapper_line"]), suffix=str(gate_cfg["suffix"]))
    gate_enabled = bool(gate_cfg.get("enabled", True))
    gate_force_on = bool(gate_cfg.get("force_on", False))
    gate_pred = boolqa_gate(prompt, gate)
    if not gate_pred:
        raise ValueError("RefBool-L prompt must be gate-true.")

    toks0 = adapter.tokenize([prompt])
    base_ids = toks0["input_ids"]  # [1, T]
    device = base_ids.device

    out: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if out:
                add = torch.tensor([out], dtype=base_ids.dtype, device=device)
                input_ids = torch.cat([base_ids, add], dim=1)
            else:
                input_ids = base_ids
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

            handles = []
            try:
                if patch is not None and patch.params:
                    pos = torch.tensor([int(fixed_prompt_pos)], dtype=torch.int64, device=device)
                    gate_mask = torch.tensor([gate_pred], dtype=torch.bool, device=device)
                    if gate_force_on:
                        gate_mask = torch.ones_like(gate_mask)
                    elif not gate_enabled:
                        gate_mask = torch.zeros_like(gate_mask)
                    for layer in cand_layers:
                        handles.append(
                            apply_hookpoint_patch(
                                adapter,
                                kind=kind,
                                layer=int(layer),
                                batch_positions=pos,
                                gate_mask=gate_mask,
                                patch_fn=_make_patch_fn(patch, layer=int(layer)),
                            )
                        )
                logits = adapter.forward_logits(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                for h in handles:
                    try:
                        h.handle.remove()
                    except Exception:  # noqa: BLE001
                        pass

            next_id = int(torch.as_tensor(logits)[0, -1, :].argmax(dim=-1).item())
            out.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
    return out


def eval_spec_exact(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch: GLRHookPatch,
    examples: Sequence[SpecExample],
) -> SpecMetrics:
    """Evaluate a spec by exact sweep (examples provided in canonical order)."""
    if not examples:
        raise ValueError("examples must be non-empty for eval_spec_exact.")

    gate_cfg = cfg.get("gate", {})
    if not isinstance(gate_cfg, Mapping):
        raise ValueError("cfg['gate'] must be a mapping.")
    gate = GateSpec(wrapper_line=str(gate_cfg["wrapper_line"]), suffix=str(gate_cfg["suffix"]))
    gate_enabled = bool(gate_cfg.get("enabled", True))
    gate_force_on = bool(gate_cfg.get("force_on", False))

    tokens = assert_or_select_answer_tokens(adapter, cfg)
    yes_id = int(tokens["yes_id"])
    no_id = int(tokens["no_id"])

    hook_cfg = cfg.get("hookpoints", {})
    if not isinstance(hook_cfg, Mapping):
        hook_cfg = {}
    kind = str(hook_cfg.get("kind", "resid_post"))

    cand_layers: list[int] = []
    if patch.params:
        cand_cfg = hook_cfg.get("candidate_layers", {})
        if not isinstance(cand_cfg, Mapping):
            raise ValueError(
                "cfg['hookpoints']['candidate_layers'] must be a mapping when patch is applied."
            )
        mode = str(cand_cfg.get("mode", "quartiles"))
        explicit = cand_cfg.get("explicit")
        cand_layers = adapter.resolve_candidate_layers(
            mode, explicit=explicit if isinstance(explicit, list) else None
        )
        missing = [layer for layer in cand_layers if layer not in patch.params]
        if missing:
            raise ValueError(f"Patch missing parameters for candidate layers: {missing}")

    batch_size = (
        int(cfg.get("evaluation", {}).get("batch_size", 64))
        if isinstance(cfg.get("evaluation"), Mapping)
        else 64
    )
    batch_size = max(1, batch_size)

    failures = 0
    margins: list[torch.Tensor] = []

    prompts_all = [e.prompt for e in examples]
    labels_all = torch.tensor([int(e.label) for e in examples], dtype=torch.int64)

    gate_pred_all = [boolqa_gate(p, gate) for p in prompts_all]
    if not all(gate_pred_all):
        bad = gate_pred_all.index(False)
        raise ValueError(f"Spec example at index {bad} is out-of-scope (gate=false).")

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            end = min(len(examples), start + batch_size)
            prompts = prompts_all[start:end]
            labels = labels_all[start:end]

            toks = adapter.tokenize(prompts)
            input_ids = toks["input_ids"]
            attention_mask = toks["attention_mask"]
            p = answer_positions(attention_mask)

            gate_mask = torch.tensor(gate_pred_all[start:end], dtype=torch.bool, device=p.device)
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
                                layer=layer,
                                batch_positions=p,
                                gate_mask=gate_mask,
                                patch_fn=_make_patch_fn(patch, layer=int(layer)),
                            )
                        )

                logits = adapter.forward_logits(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                for h in handles:
                    try:
                        h.handle.remove()
                    except Exception:  # noqa: BLE001
                        pass

            logits_p = gather_logits_at_positions(logits, p)
            yes_logits = logits_p[:, yes_id]
            no_logits = logits_p[:, no_id]

            pred = (yes_logits > no_logits).to(dtype=torch.int64)
            failures += int((pred != labels.to(device=pred.device)).sum().item())

            correct_logits = torch.where(
                labels.to(device=yes_logits.device) == 1, yes_logits, no_logits
            )
            incorrect_logits = torch.where(
                labels.to(device=yes_logits.device) == 1, no_logits, yes_logits
            )
            margins.append((correct_logits - incorrect_logits).detach().cpu())

    margins_np = torch.cat(margins, dim=0).numpy()
    total = int(len(examples))
    pass_rate = float(1.0 - (failures / total))
    min_margin = float(margins_np.min())
    p05_margin = float(np.quantile(margins_np, 0.05))

    return SpecMetrics(
        total=total,
        failures=int(failures),
        pass_rate=pass_rate,
        min_margin=min_margin,
        p05_margin=p05_margin,
    )


def eval_spec_exact_with_strata(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch: GLRHookPatch,
    examples: Sequence[SpecExample],
    strata_key: str = "stratum",
) -> tuple[SpecMetrics, dict[str, StratumMetrics]]:
    """Evaluate a spec by exact sweep and return a per-stratum breakdown.

    This is intended for coverage-bounded specs (e.g., COMPARE-6D-STRAT) where examples carry a
    deterministic `meta[strata_key]` label.
    """
    if not examples:
        raise ValueError("examples must be non-empty for eval_spec_exact_with_strata.")

    # Reuse eval_spec_exact logic for correctness; accumulate per-stratum counts in the same pass.
    gate_cfg = cfg.get("gate", {})
    if not isinstance(gate_cfg, Mapping):
        raise ValueError("cfg['gate'] must be a mapping.")
    gate = GateSpec(wrapper_line=str(gate_cfg["wrapper_line"]), suffix=str(gate_cfg["suffix"]))
    gate_enabled = bool(gate_cfg.get("enabled", True))
    gate_force_on = bool(gate_cfg.get("force_on", False))

    tokens = assert_or_select_answer_tokens(adapter, cfg)
    yes_id = int(tokens["yes_id"])
    no_id = int(tokens["no_id"])

    hook_cfg = cfg.get("hookpoints", {})
    if not isinstance(hook_cfg, Mapping):
        hook_cfg = {}
    kind = str(hook_cfg.get("kind", "resid_post"))

    cand_layers = _resolve_candidate_layers_for_patch(cfg, adapter, patch)

    batch_size = (
        int(cfg.get("evaluation", {}).get("batch_size", 64))
        if isinstance(cfg.get("evaluation"), Mapping)
        else 64
    )
    batch_size = max(1, batch_size)

    prompts_all = [e.prompt for e in examples]
    labels_all = torch.tensor([int(e.label) for e in examples], dtype=torch.int64)

    strata_all: list[str] = []
    for e in examples:
        s = None
        if isinstance(e.meta, Mapping):
            s = e.meta.get(strata_key)
        strata_all.append(str(s) if isinstance(s, str) and s else "unknown")

    gate_pred_all = [boolqa_gate(p, gate) for p in prompts_all]
    if not all(gate_pred_all):
        bad = gate_pred_all.index(False)
        raise ValueError(f"Spec example at index {bad} is out-of-scope (gate=false).")

    total_by: dict[str, int] = {}
    failures_by: dict[str, int] = {}
    for s in strata_all:
        total_by[s] = int(total_by.get(s, 0) + 1)
        failures_by.setdefault(s, 0)

    failures_total = 0
    margins: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            end = min(len(examples), start + batch_size)
            prompts = prompts_all[start:end]
            labels = labels_all[start:end]

            toks = adapter.tokenize(prompts)
            input_ids = toks["input_ids"]
            attention_mask = toks["attention_mask"]
            p = answer_positions(attention_mask)

            gate_mask = torch.tensor(gate_pred_all[start:end], dtype=torch.bool, device=p.device)
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
                                batch_positions=p,
                                gate_mask=gate_mask,
                                patch_fn=_make_patch_fn(patch, layer=int(layer)),
                            )
                        )

                logits = adapter.forward_logits(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                for h in handles:
                    try:
                        h.handle.remove()
                    except Exception:  # noqa: BLE001
                        pass

            logits_p = gather_logits_at_positions(logits, p)
            yes_logits = logits_p[:, yes_id]
            no_logits = logits_p[:, no_id]

            pred = (yes_logits > no_logits).to(dtype=torch.int64)
            mism = (pred != labels.to(device=pred.device)).detach().cpu().tolist()
            failures_total += int(sum(1 for m in mism if m))

            for i, is_fail in enumerate(mism):
                if not is_fail:
                    continue
                s = strata_all[start + i]
                failures_by[s] = int(failures_by.get(s, 0) + 1)

            correct_logits = torch.where(
                labels.to(device=yes_logits.device) == 1, yes_logits, no_logits
            )
            incorrect_logits = torch.where(
                labels.to(device=yes_logits.device) == 1, no_logits, yes_logits
            )
            margins.append((correct_logits - incorrect_logits).detach().cpu())

    margins_np = torch.cat(margins, dim=0).numpy()
    total = int(len(examples))
    pass_rate = float(1.0 - (failures_total / total))
    min_margin = float(margins_np.min())
    p05_margin = float(np.quantile(margins_np, 0.05))

    breakdown: dict[str, StratumMetrics] = {}
    for s in sorted(total_by):
        t = int(total_by[s])
        f = int(failures_by.get(s, 0))
        breakdown[s] = StratumMetrics(
            total=t, failures=f, pass_rate=float(1.0 - (f / t)) if t else 0.0
        )

    return (
        SpecMetrics(
            total=total,
            failures=int(failures_total),
            pass_rate=pass_rate,
            min_margin=min_margin,
            p05_margin=p05_margin,
        ),
        breakdown,
    )


def eval_collateral(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch: GLRHookPatch,
    refbool_s_prompts: Sequence[str],
    refbool_l_prompts: Sequence[str],
    reftext_prompts: Sequence[str],
) -> CollateralMetrics:
    """Evaluate collateral drift metrics."""
    hook_cfg = cfg.get("hookpoints", {})
    if not isinstance(hook_cfg, Mapping):
        hook_cfg = {}
    kind = str(hook_cfg.get("kind", "resid_post"))

    cand_layers = _resolve_candidate_layers_for_patch(cfg, adapter, patch)

    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), Mapping) else {}
    batch_size = int(eval_cfg.get("batch_size", 64))
    batch_size = max(1, batch_size)
    resamples = int(eval_cfg.get("bootstrap_resamples", 2000))
    bootstrap_seed = int(eval_cfg.get("bootstrap_seed", 0))

    s_mean, s_values = _mean_kl_at_answer_position(
        cfg=cfg,
        adapter=adapter,
        patch=patch,
        prompts=refbool_s_prompts,
        require_gate=True,
        kind=kind,
        cand_layers=cand_layers,
        batch_size=batch_size,
    )
    ci_lo, ci_hi = _bootstrap_ci95(s_values, resamples=resamples, seed=bootstrap_seed)

    t_mean, _t_values = _mean_kl_at_answer_position(
        cfg=cfg,
        adapter=adapter,
        patch=patch,
        prompts=reftext_prompts,
        require_gate=False,
        kind=kind,
        cand_layers=cand_layers,
        batch_size=batch_size,
    )

    # RefBool-L: greedy generation drift on continuation tokens.
    if refbool_l_prompts:
        gen_cfg = (
            eval_cfg.get("generation", {})
            if isinstance(eval_cfg.get("generation"), Mapping)
            else {}
        )
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 128))
        max_new_tokens = max(0, max_new_tokens)

        divergence = 0
        first_diffs: list[int] = []
        edit_norms: list[float] = []

        for p in refbool_l_prompts:
            toks = adapter.tokenize([p])
            p_idx = int(answer_positions(toks["attention_mask"])[0].item())
            base = _generate_greedy_continuation(
                cfg=cfg,
                adapter=adapter,
                patch=None,
                prompt=p,
                kind=kind,
                cand_layers=cand_layers,
                fixed_prompt_pos=p_idx,
            )
            patched = _generate_greedy_continuation(
                cfg=cfg,
                adapter=adapter,
                patch=patch,
                prompt=p,
                kind=kind,
                cand_layers=cand_layers,
                fixed_prompt_pos=p_idx,
            )
            if base != patched:
                divergence += 1
            first = _first_diff_index(base, patched, default=max_new_tokens)
            first_diffs.append(int(first))
            dist = _token_edit_distance(base, patched)
            denom = max(len(base), len(patched), 1)
            edit_norms.append(float(dist / denom))

        refbool_l_div = float(divergence / max(1, len(refbool_l_prompts)))
        refbool_l_first = (
            float(np.mean(np.asarray(first_diffs, dtype=np.float64))) if first_diffs else 0.0
        )
        refbool_l_edit = (
            float(np.mean(np.asarray(edit_norms, dtype=np.float64))) if edit_norms else 0.0
        )
    else:
        refbool_l_div = 0.0
        refbool_l_first = 0.0
        refbool_l_edit = 0.0

    return CollateralMetrics(
        refbool_s_mean_kl=float(s_mean),
        refbool_s_ci95=(float(ci_lo), float(ci_hi)),
        refbool_l_divergence_rate=float(refbool_l_div),
        refbool_l_first_diff_index=float(refbool_l_first),
        refbool_l_norm_edit_distance=float(refbool_l_edit),
        reftext_mean_kl=float(t_mean),
    )
