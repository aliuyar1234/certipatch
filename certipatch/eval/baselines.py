"""certipatch.eval.baselines

Baselines are defined in SSOT:
- `07_Baselines_Ablations.md`
- `EXPERIMENTS.md`

This module is used by `scripts/reproduce_paper.py` to populate `runs/<run_id>/baselines.json`.
Only CertiPatch (GLR-HP) runs are certificate-verified; baselines are comparative metrics only.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch

from certipatch.cegis.trainer import SolverState, solve_constrained_minimality, solve_multiobjective
from certipatch.data_generation import build_refbool_l, build_refbool_s, build_reftext
from certipatch.eval.metrics import eval_collateral, eval_spec_exact
from certipatch.hooks import GateSpec, answer_positions, boolqa_gate, gather_logits_at_positions
from certipatch.models.load_model import ModelAdapter, assert_or_select_answer_tokens
from certipatch.patch_families import GLRHookPatch, GLRHPConfig
from certipatch.specs import SpecExample, SpecId


@dataclass(frozen=True)
class BaselineResult:
    name: str
    artifacts_dir: str
    metrics: Dict[str, Any]


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _alpha_tag(alpha: float) -> str:
    s = f"{float(alpha):.12f}".rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    return s.replace("-", "m").replace(".", "p")


def _checkpoint_root(cfg: Mapping[str, Any]) -> Optional[Path]:
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), Mapping) else {}
    run_id = str(run_cfg.get("run_id", "")).strip()
    if not run_id:
        return None
    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), Mapping) else {}
    out_dir = Path(str(out_cfg.get("out_dir", "runs")))
    return out_dir / run_id / "_baselines_ckpt"


def _baseline_checkpoint_path(*, ckpt_root: Path, baseline_name: str) -> Path:
    return ckpt_root / f"{baseline_name}.json"


def _alpha_checkpoint_path(*, ckpt_root: Path, baseline_name: str, alpha: float) -> Path:
    return ckpt_root / baseline_name / f"alpha_{_alpha_tag(alpha)}.json"


def _baseline_solver_checkpoint_path(*, ckpt_root: Path, baseline_name: str) -> Path:
    return ckpt_root / f"{baseline_name}.solver.pt"


def _alpha_solver_checkpoint_path(*, ckpt_root: Path, baseline_name: str, alpha: float) -> Path:
    return ckpt_root / baseline_name / f"alpha_{_alpha_tag(alpha)}.solver.pt"


def _load_baseline_checkpoint(path: Path) -> Optional[BaselineResult]:
    try:
        raw = _read_json(path)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, Mapping):
        return None
    name = raw.get("name")
    artifacts_dir = raw.get("artifacts_dir")
    metrics = raw.get("metrics")
    if (
        not isinstance(name, str)
        or not isinstance(artifacts_dir, str)
        or not isinstance(metrics, dict)
    ):
        return None
    return BaselineResult(name=name, artifacts_dir=artifacts_dir, metrics=dict(metrics))


def _normalize_trial(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return None
    return dict(raw)


def _load_trial_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = _read_json(path)
    except Exception:  # noqa: BLE001
        return None
    return _normalize_trial(raw)


def _write_torch_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(path)


def _load_torch_checkpoint(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        raw = torch.load(path, map_location="cpu")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, Mapping):
        return None
    return raw


def _solver_state_to_payload(state: Optional[SolverState]) -> Optional[Dict[str, Any]]:
    if state is None:
        return None
    return {
        "lambda_mult": float(state.lambda_mult),
        "mu": float(state.mu),
        "inner_round": int(state.inner_round),
    }


def _solver_state_from_payload(raw: Any) -> Optional[SolverState]:
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


def _ensure_runtime_prompt_suites(
    *,
    cfg: Mapping[str, Any],
    specs_enabled: Sequence[str],
    refbool_s: Sequence[str],
    refbool_l: Sequence[str],
    reftext: Sequence[str],
) -> Tuple[list[str], list[str], list[str]]:
    out_ref_s = [str(x) for x in refbool_s]
    out_ref_l = [str(x) for x in refbool_l]
    out_ref_t = [str(x) for x in reftext]

    if out_ref_s and out_ref_l and out_ref_t:
        return out_ref_s, out_ref_l, out_ref_t
    if not specs_enabled:
        return out_ref_s, out_ref_l, out_ref_t

    spec_id: SpecId = cast(SpecId, str(specs_enabled[0]))
    examples = _iter_domain_examples(cfg, spec_id)
    spec_prompt_set = {e.prompt for e in examples}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), Mapping) else {}
    n_s = int(data_cfg.get("refbool_s_n", 20000))
    n_l = int(data_cfg.get("refbool_l_n", 1000))
    n_t = int(data_cfg.get("reftext_n", 5000))

    if not out_ref_s:
        out_ref_s = [
            str(x) for x in build_refbool_s(cfg=cfg, n_prompts=n_s, spec_prompt_set=spec_prompt_set)
        ]
    if not out_ref_l:
        out_ref_l = [
            str(x) for x in build_refbool_l(cfg=cfg, n_prompts=n_l, spec_prompt_set=spec_prompt_set)
        ]
    if not out_ref_t:
        out_ref_t = [str(x) for x in build_reftext(cfg=cfg, n_prompts=n_t)]

    runtime = cfg.get("_certipatch_runtime")
    if isinstance(runtime, dict):
        runtime["refbool_s_prompts"] = list(out_ref_s)
        runtime["refbool_l_prompts"] = list(out_ref_l)
        runtime["reftext_prompts"] = list(out_ref_t)

    return out_ref_s, out_ref_l, out_ref_t


def _checkpoint_step_interval(cfg: Mapping[str, Any], default: int = 100) -> int:
    runtime = cfg.get("_certipatch_runtime")
    if not isinstance(runtime, Mapping):
        return int(max(1, int(default)))
    ckpt_cfg = runtime.get("checkpoint")
    if not isinstance(ckpt_cfg, Mapping):
        return int(max(1, int(default)))
    raw = ckpt_cfg.get("step_every", ckpt_cfg.get("solver_step_every", default))
    try:
        return int(max(1, int(raw)))
    except Exception:  # noqa: BLE001
        return int(max(1, int(default)))


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


def _resolve_candidate_layers(cfg: Mapping[str, Any], adapter: ModelAdapter) -> list[int]:
    hook_cfg = cfg.get("hookpoints", {}) if isinstance(cfg.get("hookpoints"), Mapping) else {}
    cand_cfg = (
        hook_cfg.get("candidate_layers", {})
        if isinstance(hook_cfg.get("candidate_layers"), Mapping)
        else {}
    )
    mode = str(cand_cfg.get("mode", "quartiles"))
    explicit = cand_cfg.get("explicit")
    return adapter.resolve_candidate_layers(
        mode, explicit=explicit if isinstance(explicit, list) else None
    )


def _make_glr_patch(cfg: Mapping[str, Any], *, cand_layers: Sequence[int]) -> GLRHookPatch:
    patch_cfg = cfg.get("patch", {}) if isinstance(cfg.get("patch"), Mapping) else {}
    rank_r = int(patch_cfg.get("rank_r", 4))
    threshold = float(patch_cfg.get("effective_layer_threshold", 0.001))
    return GLRHookPatch(
        cfg=GLRHPConfig(
            rank_r=rank_r,
            candidate_layers=[int(x) for x in cand_layers],
            effective_layer_threshold=threshold,
        )
    )


def _glrhp_budget(
    adapter: ModelAdapter, *, cand_layers: Sequence[int], rank_r: int
) -> Dict[str, Any]:
    d = int(adapter.info.d_model)
    L = int(len(list(cand_layers)))
    r = int(rank_r)
    budget = int(2 * d * r * L)
    return {
        "P_GLRHP": budget,
        "budget_lo": int(math.floor(0.9 * budget)),
        "budget_hi": int(math.ceil(1.1 * budget)),
    }


def _stable_int_seed(*parts: str) -> int:
    h = sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return int.from_bytes(h.digest()[:8], "little", signed=False) % (2**31 - 1)


class _SteeringVec1LPatch:
    """Patch-like object compatible with eval_spec_exact/eval_collateral (answer-position only)."""

    def __init__(self, *, layer: int, delta: torch.Tensor) -> None:
        self.layer = int(layer)
        self.delta = delta
        self.params: Dict[int, Dict[str, Any]] = {self.layer: {"delta": self.delta}}

    def apply_to_vectors(self, h: Any, layer: int) -> Any:  # noqa: ARG002
        h_t = torch.as_tensor(h)
        delta = torch.as_tensor(self.delta, device=h_t.device, dtype=h_t.dtype)
        return h_t + delta.unsqueeze(0)

    def parameter_count(self) -> int:
        return int(torch.as_tensor(self.delta).numel())

    def fro_norm(self) -> float:
        d = torch.as_tensor(self.delta, dtype=torch.float32)
        return float(d.pow(2).sum().sqrt().item())

    def serialize(self) -> Dict[str, Any]:
        return {
            "family": "SteeringVec-1L",
            "layer": int(self.layer),
            "parameter_count": self.parameter_count(),
            "fro_norm": self.fro_norm(),
        }


def _cfg_with_explicit_candidate_layers(
    cfg: Mapping[str, Any], layers: Sequence[int]
) -> Dict[str, Any]:
    out: Dict[str, Any] = copy.deepcopy(dict(cfg))
    out.setdefault("hookpoints", {})
    if not isinstance(out["hookpoints"], dict):
        out["hookpoints"] = {}
    out["hookpoints"].setdefault("candidate_layers", {})
    if not isinstance(out["hookpoints"]["candidate_layers"], dict):
        out["hookpoints"]["candidate_layers"] = {}
    out["hookpoints"]["candidate_layers"]["mode"] = "explicit"
    out["hookpoints"]["candidate_layers"]["explicit"] = [int(x) for x in layers]
    return out


def _train_steering_vec_1l(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    examples: Sequence[SpecExample],
    layer: int,
) -> Tuple[_SteeringVec1LPatch, Dict[str, Any]]:
    if not examples:
        raise ValueError("examples must be non-empty.")

    tokens = assert_or_select_answer_tokens(adapter, cfg)
    yes_id = int(tokens["yes_id"])
    no_id = int(tokens["no_id"])

    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), Mapping) else {}
    seeds = run_cfg.get("seeds", {}) if isinstance(run_cfg.get("seeds"), Mapping) else {}
    master_seed = int(seeds.get("master", 0))

    objective_cfg = cfg.get("objective", {}) if isinstance(cfg.get("objective"), Mapping) else {}
    tau = float(objective_cfg.get("tau_margin", 1.0))
    beta = float(objective_cfg.get("beta_smooth", 50.0))

    optimizer_cfg = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer"), Mapping) else {}
    steps = int(optimizer_cfg.get("inner_steps_per_outer", 2000))
    steps = max(1, steps)
    lr_grid = optimizer_cfg.get("lr_grid")
    lr = (
        float(lr_grid[len(lr_grid) // 2])
        if isinstance(lr_grid, list) and lr_grid
        else float(optimizer_cfg.get("lr", 3e-3))
    )

    betas = optimizer_cfg.get("adam_betas", [0.9, 0.999])
    if not (isinstance(betas, list) and len(betas) == 2):
        raise ValueError("cfg['optimizer']['adam_betas'] must be a list of two floats.")
    adam_betas = (float(betas[0]), float(betas[1]))

    eval_batch_size = (
        int(cfg.get("evaluation", {}).get("batch_size", 64))
        if isinstance(cfg.get("evaluation"), Mapping)
        else 64
    )
    batch_size = max(1, int(optimizer_cfg.get("spec_batch_size", eval_batch_size)))

    device = adapter.tokenize([examples[0].prompt])["input_ids"].device
    delta = torch.nn.Parameter(
        torch.zeros((int(adapter.info.d_model),), device=device, dtype=torch.float32)
    )
    opt = torch.optim.Adam([delta], lr=lr, betas=adam_betas)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(master_seed + 202)
    perm = torch.randperm(len(examples), generator=gen).tolist()
    pos = 0

    hook_cfg = cfg.get("hookpoints", {}) if isinstance(cfg.get("hookpoints"), Mapping) else {}
    kind = str(hook_cfg.get("kind", "resid_post"))

    gate_cfg = cfg.get("gate", {}) if isinstance(cfg.get("gate"), Mapping) else {}
    gate = GateSpec(
        wrapper_line=str(gate_cfg.get("wrapper_line", "")), suffix=str(gate_cfg.get("suffix", ""))
    )
    gate_enabled = bool(gate_cfg.get("enabled", True))
    gate_force_on = bool(gate_cfg.get("force_on", False))

    from certipatch.hooks import apply_hookpoint_patch

    last_g = 0.0
    for _ in range(steps):
        ids = [perm[(pos + i) % len(perm)] for i in range(batch_size)]
        pos = (pos + batch_size) % len(perm)
        batch = [examples[i] for i in ids]

        prompts = [e.prompt for e in batch]
        labels = torch.tensor([int(e.label) for e in batch], dtype=torch.int64, device=device)

        gate_pred = [boolqa_gate(p, gate) for p in prompts]
        if not all(gate_pred):
            bad = gate_pred.index(False)
            raise ValueError(f"Spec prompt out-of-scope for steering training at index {bad}.")

        toks = adapter.tokenize(prompts)
        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]
        p_idx = answer_positions(attention_mask)

        gate_mask = torch.tensor(gate_pred, dtype=torch.bool, device=p_idx.device)
        if gate_force_on:
            gate_mask = torch.ones_like(gate_mask)
        elif not gate_enabled:
            gate_mask = torch.zeros_like(gate_mask)

        opt.zero_grad(set_to_none=True)
        handles = []
        try:
            handles.append(
                apply_hookpoint_patch(
                    adapter,
                    kind=kind,
                    layer=int(layer),
                    batch_positions=p_idx,
                    gate_mask=gate_mask,
                    patch_fn=lambda h: torch.as_tensor(h)
                    + delta.to(device=h.device, dtype=h.dtype).unsqueeze(0),
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
        yes = logits_p[:, yes_id]
        no = logits_p[:, no_id]
        correct = torch.where(labels == 1, yes, no)
        incorrect = torch.where(labels == 1, no, yes)
        margins = correct - incorrect

        v = torch.relu(torch.tensor(tau, device=margins.device) - margins)
        g_smooth = (torch.logsumexp(beta * v, dim=0) - math.log(max(1, v.numel()))) / beta
        if not torch.isfinite(g_smooth):
            raise ValueError("Non-finite steering loss.")

        g_smooth.backward()
        opt.step()
        last_g = float(g_smooth.detach().cpu().item())

    patch = _SteeringVec1LPatch(layer=int(layer), delta=delta.detach())
    diag = {
        "layer": int(layer),
        "steps": int(steps),
        "lr": float(lr),
        "g_smooth_last": float(last_g),
    }
    return patch, diag


def _eval_all_specs_glr(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch: GLRHookPatch,
    refbool_s_prompts: Sequence[str],
    refbool_l_prompts: Sequence[str],
    reftext_prompts: Sequence[str],
) -> Dict[str, Any]:
    specs_cfg = cfg.get("specs", {}) if isinstance(cfg.get("specs"), Mapping) else {}
    enabled = specs_cfg.get("enabled", [])
    if not isinstance(enabled, list) or not all(isinstance(s, str) for s in enabled):
        enabled = []

    spec_metrics: dict[str, Any] = {}
    for sid in enabled:
        examples = _iter_domain_examples(cfg, cast(SpecId, sid))
        spec_metrics[str(sid)] = eval_spec_exact(
            cfg=cfg, adapter=adapter, patch=patch, examples=examples
        ).__dict__

    col = eval_collateral(
        cfg=cfg,
        adapter=adapter,
        patch=patch,
        refbool_s_prompts=refbool_s_prompts,
        refbool_l_prompts=refbool_l_prompts,
        reftext_prompts=reftext_prompts,
    )
    return {
        "spec_metrics": spec_metrics,
        "collateral_metrics": col.__dict__,
        "patch": patch.serialize(),
    }


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


def _kl_pq_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits_p.float(), dim=-1)
    logq = torch.log_softmax(logits_q.float(), dim=-1)
    p = logp.exp()
    return (p * (logp - logq)).sum(dim=-1)


def _softprompt_forward_logits_hf(
    *,
    adapter: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    softprompt: torch.Tensor,
    use_cache: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    model = getattr(adapter, "model", None)
    if model is None:
        raise ValueError("SoftPrompt baseline requires HuggingFaceAdapter with `.model`.")

    emb = model.get_input_embeddings()(input_ids)
    k = int(softprompt.shape[0])
    prefix = (
        softprompt.to(device=emb.device, dtype=emb.dtype)
        .unsqueeze(0)
        .expand(emb.shape[0], k, emb.shape[2])
    )
    inputs_embeds = torch.cat([prefix, emb], dim=1)
    attn2 = torch.cat(
        [
            torch.ones((emb.shape[0], k), device=attention_mask.device, dtype=attention_mask.dtype),
            attention_mask,
        ],
        dim=1,
    )
    out = model(inputs_embeds=inputs_embeds, attention_mask=attn2, use_cache=bool(use_cache))
    logits = getattr(out, "logits", None)
    past = getattr(out, "past_key_values", None)
    if logits is None:
        raise ValueError("HF model forward did not return logits for SoftPrompt baseline.")
    return torch.as_tensor(logits), attn2, past


def _eval_spec_exact_softprompt_hf(
    *,
    cfg: Mapping[str, Any],
    adapter: Any,
    softprompt: torch.Tensor,
    examples: Sequence[SpecExample],
) -> Dict[str, Any]:
    if not examples:
        raise ValueError("examples must be non-empty.")

    gate_cfg = cfg.get("gate", {}) if isinstance(cfg.get("gate"), Mapping) else {}
    gate = GateSpec(wrapper_line=str(gate_cfg["wrapper_line"]), suffix=str(gate_cfg["suffix"]))
    if not all(boolqa_gate(e.prompt, gate) for e in examples):
        raise ValueError("Spec examples must be gate-true for SoftPrompt baseline.")

    tokens = assert_or_select_answer_tokens(adapter, cfg)
    yes_id = int(tokens["yes_id"])
    no_id = int(tokens["no_id"])

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

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            end = min(len(examples), start + batch_size)
            prompts = prompts_all[start:end]
            labels = labels_all[start:end]

            toks = adapter.tokenize(prompts)
            logits, attn2, _past = _softprompt_forward_logits_hf(
                adapter=adapter,
                input_ids=toks["input_ids"],
                attention_mask=toks["attention_mask"],
                softprompt=softprompt,
                use_cache=False,
            )
            p = answer_positions(attn2)
            logits_p = gather_logits_at_positions(logits, p)
            yes = logits_p[:, yes_id]
            no = logits_p[:, no_id]

            pred = (yes > no).to(dtype=torch.int64)
            failures += int((pred != labels.to(device=pred.device)).sum().item())

            correct = torch.where(labels.to(device=yes.device) == 1, yes, no)
            incorrect = torch.where(labels.to(device=yes.device) == 1, no, yes)
            margins.append((correct - incorrect).detach().cpu())

    margins_np = torch.cat(margins, dim=0).numpy()
    total = int(len(examples))
    return {
        "total": total,
        "failures": int(failures),
        "pass_rate": float(1.0 - (failures / total)),
        "min_margin": float(margins_np.min()),
        "p05_margin": float(np.quantile(margins_np, 0.05)),
    }


def _eval_collateral_softprompt_hf(
    *,
    cfg: Mapping[str, Any],
    adapter: Any,
    softprompt: torch.Tensor,
    refbool_s_prompts: Sequence[str],
    refbool_l_prompts: Sequence[str],
    reftext_prompts: Sequence[str],
) -> Dict[str, Any]:
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), Mapping) else {}
    batch_size = max(1, int(eval_cfg.get("batch_size", 64)))
    resamples = int(eval_cfg.get("bootstrap_resamples", 2000))
    bootstrap_seed = int(eval_cfg.get("bootstrap_seed", 0))
    gen_cfg = (
        eval_cfg.get("generation", {}) if isinstance(eval_cfg.get("generation"), Mapping) else {}
    )
    max_new_tokens = max(0, int(gen_cfg.get("max_new_tokens", 128)))

    gate_cfg = cfg.get("gate", {}) if isinstance(cfg.get("gate"), Mapping) else {}
    gate = GateSpec(wrapper_line=str(gate_cfg["wrapper_line"]), suffix=str(gate_cfg["suffix"]))
    gate_enabled = bool(gate_cfg.get("enabled", True))

    # RefBool-S KL (gate-true prompts; softprompt applies only when gate is enabled).
    if not all(boolqa_gate(p, gate) for p in refbool_s_prompts):
        raise ValueError("RefBool-S prompts must be gate-true.")

    kls: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(refbool_s_prompts), batch_size):
            end = min(len(refbool_s_prompts), start + batch_size)
            batch = list(refbool_s_prompts[start:end])
            toks = adapter.tokenize(batch)
            base_logits = adapter.forward_logits(
                input_ids=toks["input_ids"], attention_mask=toks["attention_mask"]
            )
            p_base = answer_positions(toks["attention_mask"])
            base_p = gather_logits_at_positions(base_logits, p_base)

            if gate_enabled:
                logits, attn2, _past = _softprompt_forward_logits_hf(
                    adapter=adapter,
                    input_ids=toks["input_ids"],
                    attention_mask=toks["attention_mask"],
                    softprompt=softprompt,
                    use_cache=False,
                )
                p_patch = answer_positions(attn2)
                patch_p = gather_logits_at_positions(logits, p_patch)
                kl = _kl_pq_from_logits(base_p, patch_p).detach().cpu().numpy().astype(np.float64)
            else:
                kl = np.zeros((end - start,), dtype=np.float64)
            kls.append(kl)

    kl_all = np.concatenate(kls, axis=0) if kls else np.zeros((0,), dtype=np.float64)
    s_mean = float(kl_all.mean()) if kl_all.size else 0.0
    ci_lo, ci_hi = _bootstrap_ci95(kl_all, resamples=resamples, seed=bootstrap_seed)

    # RefText KL: out-of-scope prompts, so softprompt must not apply.
    if any(boolqa_gate(p, gate) for p in reftext_prompts):
        raise ValueError("RefText prompts must be gate-false.")
    t_mean = 0.0

    # RefBool-L drift via cached greedy generation; softprompt used only for prompt forward.
    model = getattr(adapter, "model", None)
    if model is None:
        raise ValueError("SoftPrompt baseline requires HuggingFaceAdapter with `.model`.")

    eos_id = getattr(getattr(adapter, "tokenizer", None), "eos_token_id", None)
    eos_id = int(eos_id) if eos_id is not None else None

    divergence = 0
    first_diffs: list[int] = []
    edit_norms: list[float] = []

    for p in refbool_l_prompts:
        if not boolqa_gate(p, gate):
            raise ValueError("RefBool-L prompts must be gate-true.")

        toks = adapter.tokenize([p])
        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]

        with torch.no_grad():
            out0 = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            logits0 = getattr(out0, "logits", None)
            past0 = getattr(out0, "past_key_values", None)
            if logits0 is None or past0 is None:
                raise ValueError("Base RefBool-L prompt forward missing logits/past_key_values.")

            if gate_enabled:
                logits1, _attn2, past1 = _softprompt_forward_logits_hf(
                    adapter=adapter,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    softprompt=softprompt,
                    use_cache=True,
                )
                if past1 is None:
                    raise ValueError("SoftPrompt prompt forward missing past_key_values.")
            else:
                logits1 = torch.as_tensor(logits0)
                past1 = past0

            base_next = int(torch.as_tensor(logits0)[0, -1, :].argmax(dim=-1).item())
            patch_next = int(torch.as_tensor(logits1)[0, -1, :].argmax(dim=-1).item())

            base_tokens: list[int] = [base_next]
            patch_tokens: list[int] = [patch_next]

            base_tok = torch.tensor([[base_next]], device=input_ids.device, dtype=input_ids.dtype)
            patch_tok = torch.tensor([[patch_next]], device=input_ids.device, dtype=input_ids.dtype)

            for _ in range(max_new_tokens - 1):
                out_b = model(input_ids=base_tok, past_key_values=past0, use_cache=True)
                logits_b = getattr(out_b, "logits", None)
                past0 = getattr(out_b, "past_key_values", None)
                if logits_b is None or past0 is None:
                    raise ValueError("Base RefBool-L continuation missing logits/past_key_values.")
                base_next = int(torch.as_tensor(logits_b)[0, -1, :].argmax(dim=-1).item())
                base_tokens.append(base_next)
                base_tok = torch.tensor(
                    [[base_next]], device=input_ids.device, dtype=input_ids.dtype
                )

                # Softprompt disabled after prompt forward; use base weights with patched cache.
                out_p = model(input_ids=patch_tok, past_key_values=past1, use_cache=True)
                logits_p = getattr(out_p, "logits", None)
                past1 = getattr(out_p, "past_key_values", None)
                if logits_p is None or past1 is None:
                    raise ValueError(
                        "Patched RefBool-L continuation missing logits/past_key_values."
                    )
                patch_next = int(torch.as_tensor(logits_p)[0, -1, :].argmax(dim=-1).item())
                patch_tokens.append(patch_next)
                patch_tok = torch.tensor(
                    [[patch_next]], device=input_ids.device, dtype=input_ids.dtype
                )

                if eos_id is not None and (base_next == eos_id or patch_next == eos_id):
                    break

        if base_tokens != patch_tokens:
            divergence += 1
        # token-level edit + first diff
        k = min(len(base_tokens), len(patch_tokens))
        first = next(
            (i for i in range(k) if base_tokens[i] != patch_tokens[i]),
            k if len(base_tokens) != len(patch_tokens) else max_new_tokens,
        )
        first_diffs.append(int(first))
        dist = _token_edit_distance(base_tokens, patch_tokens)
        denom = max(len(base_tokens), len(patch_tokens), 1)
        edit_norms.append(float(dist / denom))

    div_rate = float(divergence / max(1, len(refbool_l_prompts))) if refbool_l_prompts else 0.0
    mean_first = float(np.mean(np.asarray(first_diffs, dtype=np.float64))) if first_diffs else 0.0
    mean_edit = float(np.mean(np.asarray(edit_norms, dtype=np.float64))) if edit_norms else 0.0

    return {
        "refbool_s_mean_kl": float(s_mean),
        "refbool_s_ci95": (float(ci_lo), float(ci_hi)),
        "refbool_l_divergence_rate": float(div_rate),
        "refbool_l_first_diff_index": float(mean_first),
        "refbool_l_norm_edit_distance": float(mean_edit),
        "reftext_mean_kl": float(t_mean),
    }


def _token_edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
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


def _train_softprompt_hf(
    *,
    cfg: Mapping[str, Any],
    adapter: Any,
    examples: Sequence[SpecExample],
    refbool_s_prompts: Sequence[str],
    k_virtual_tokens: int,
    alpha: float,
    seed_offset: int,
    init_softprompt: Optional[torch.Tensor] = None,
    resume_step: int = 0,
    on_step_end: Optional[Callable[[torch.Tensor, Mapping[str, Any]], None]] = None,
    checkpoint_every: int = 100,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if not examples:
        raise ValueError("examples must be non-empty.")
    if not refbool_s_prompts:
        raise ValueError("refbool_s_prompts must be non-empty.")

    model = getattr(adapter, "model", None)
    if model is None:
        raise ValueError("SoftPrompt baseline requires HuggingFaceAdapter with `.model`.")

    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), Mapping) else {}
    seeds = run_cfg.get("seeds", {}) if isinstance(run_cfg.get("seeds"), Mapping) else {}
    master_seed = int(seeds.get("master", 0))

    optimizer_cfg = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer"), Mapping) else {}
    lr_grid = optimizer_cfg.get("lr_grid")
    lr = (
        float(lr_grid[len(lr_grid) // 2])
        if isinstance(lr_grid, list) and lr_grid
        else float(optimizer_cfg.get("lr", 3e-3))
    )
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

    objective_cfg = cfg.get("objective", {}) if isinstance(cfg.get("objective"), Mapping) else {}
    tau = float(objective_cfg.get("tau_margin", 1.0))
    beta = float(objective_cfg.get("beta_smooth", 50.0))

    tokens = assert_or_select_answer_tokens(adapter, cfg)
    yes_id = int(tokens["yes_id"])
    no_id = int(tokens["no_id"])

    device = adapter.tokenize([examples[0].prompt])["input_ids"].device
    d_model = int(adapter.info.d_model)

    if init_softprompt is not None:
        soft_init = torch.as_tensor(init_softprompt, dtype=torch.float32).detach().clone()
        if soft_init.shape != (int(k_virtual_tokens), d_model):
            raise ValueError(
                "init_softprompt shape mismatch:"
                f" expected {(int(k_virtual_tokens), d_model)}, got {tuple(soft_init.shape)}"
            )
    else:
        # Initialize on CPU with an explicit CPU generator, then move to `device`.
        # (PyTorch requires the generator's device type to match the tensor's device.)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(master_seed + 7000 + int(seed_offset))
        soft_init = (
            torch.randn(
                (int(k_virtual_tokens), d_model),
                generator=gen,
                device="cpu",
                dtype=torch.float32,
            )
            * 0.01
        )
    soft = torch.nn.Parameter(soft_init.to(device=device))
    opt = torch.optim.Adam([soft], lr=lr, betas=adam_betas)

    perm_gen = torch.Generator(device="cpu")
    perm_gen.manual_seed(master_seed + 8000 + int(seed_offset))
    spec_perm = torch.randperm(len(examples), generator=perm_gen).tolist()
    ref_perm = torch.randperm(len(refbool_s_prompts), generator=perm_gen).tolist()
    spec_pos = (start_step * spec_batch_size) % len(spec_perm)
    ref_pos = (start_step * ref_batch_size) % len(ref_perm)

    gate_cfg = cfg.get("gate", {}) if isinstance(cfg.get("gate"), Mapping) else {}
    gate_enabled = bool(gate_cfg.get("enabled", True))

    last_loss = 0.0
    for _step in range(start_step, steps):
        s_ids = [spec_perm[(spec_pos + i) % len(spec_perm)] for i in range(spec_batch_size)]
        spec_pos = (spec_pos + spec_batch_size) % len(spec_perm)
        r_ids = [ref_perm[(ref_pos + i) % len(ref_perm)] for i in range(ref_batch_size)]
        ref_pos = (ref_pos + ref_batch_size) % len(ref_perm)

        spec_batch = [examples[i] for i in s_ids]
        ref_batch = [refbool_s_prompts[i] for i in r_ids]

        opt.zero_grad(set_to_none=True)

        # Collateral KL on RefBool-S minibatch.
        if gate_enabled and float(alpha) != 0.0:
            toks_ref = adapter.tokenize(ref_batch)
            base_logits = adapter.forward_logits(
                input_ids=toks_ref["input_ids"], attention_mask=toks_ref["attention_mask"]
            )
            p_base = answer_positions(toks_ref["attention_mask"])
            base_p = gather_logits_at_positions(base_logits, p_base)

            patch_logits, attn2, _past = _softprompt_forward_logits_hf(
                adapter=adapter,
                input_ids=toks_ref["input_ids"],
                attention_mask=toks_ref["attention_mask"],
                softprompt=soft,
                use_cache=False,
            )
            p_patch = answer_positions(attn2)
            patch_p = gather_logits_at_positions(patch_logits, p_patch)
            kl = _kl_pq_from_logits(base_p, patch_p).mean()
        else:
            kl = torch.tensor(0.0, device=device, dtype=torch.float32)

        # Spec smooth violation proxy on a spec minibatch.
        prompts = [e.prompt for e in spec_batch]
        labels = torch.tensor([int(e.label) for e in spec_batch], dtype=torch.int64, device=device)
        toks_spec = adapter.tokenize(prompts)
        logits_s, attn2_s, _past_s = _softprompt_forward_logits_hf(
            adapter=adapter,
            input_ids=toks_spec["input_ids"],
            attention_mask=toks_spec["attention_mask"],
            softprompt=soft,
            use_cache=False,
        )
        p_s = answer_positions(attn2_s)
        logits_s_p = gather_logits_at_positions(logits_s, p_s)
        yes = logits_s_p[:, yes_id]
        no = logits_s_p[:, no_id]
        correct = torch.where(labels == 1, yes, no)
        incorrect = torch.where(labels == 1, no, yes)
        margins = correct - incorrect
        v = torch.relu(torch.tensor(tau, device=margins.device) - margins)
        g_smooth = (torch.logsumexp(beta * v, dim=0) - math.log(max(1, v.numel()))) / beta

        loss = g_smooth + float(alpha) * kl
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss in SoftPrompt training.")

        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu().item())

        if on_step_end is not None and (
            _step == start_step or (_step + 1) % int(max(1, checkpoint_every)) == 0 or _step + 1 == steps
        ):
            on_step_end(
                soft.detach(),
                {
                    "step_next": int(_step + 1),
                    "steps": int(steps),
                    "alpha": float(alpha),
                    "loss": float(last_loss),
                },
            )

    diag = {
        "alpha": float(alpha),
        "steps": int(steps),
        "resume_step": int(start_step),
        "steps_run": int(max(0, steps - start_step)),
        "lr": float(lr),
        "loss_last": float(last_loss),
        "param_count": int(soft.numel()),
        "fro_norm": float(
            torch.as_tensor(soft.detach(), dtype=torch.float32).pow(2).sum().sqrt().item()
        ),
    }
    return soft.detach(), diag


class _LoRAState:
    def __init__(self) -> None:
        self.enabled: bool = False
        self.params: list[torch.nn.Parameter] = []
        self.handles: list[Any] = []
        self.layers: list[int] = []
        self.target_modules: list[str] = []
        self.resolved_modules: list[str] = []

    def parameter_count(self) -> int:
        return int(sum(int(p.numel()) for p in self.params))

    def fro_norm(self) -> float:
        if not self.params:
            return 0.0
        total = torch.tensor(0.0, device=self.params[0].device)
        for p in self.params:
            total = total + torch.as_tensor(p, dtype=torch.float32).pow(2).sum()
        return float(total.sqrt().item())

    def remove(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self.handles = []


def _hf_blocks(model: Any) -> Any:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Unsupported HF model structure: cannot locate transformer block list.")


def _resolve_attr_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in str(path).split("."):
        if not hasattr(cur, part):
            raise ValueError(f"Module path not found: {path} (missing {part})")
        cur = getattr(cur, part)
    return cur


def _resolve_attr_path_any(obj: Any, path_spec: str) -> tuple[Any, str]:
    """Resolve a dotted attribute path, supporting ordered alternatives separated by '|'."""
    spec = str(path_spec)
    options = [p.strip() for p in spec.split("|") if p.strip()]
    if not options:
        options = [spec]

    last_err: Optional[Exception] = None
    for p in options:
        try:
            return _resolve_attr_path(obj, p), p
        except Exception as e:  # noqa: BLE001 - surfaced as fail-closed ValueError below
            last_err = e
            continue

    detail = f"{last_err}" if last_err is not None else "unknown"
    raise ValueError(f"Module path not found: {spec} (tried {options}): {detail}")


def _install_lora_hf(
    *,
    cfg: Mapping[str, Any],
    adapter: Any,
    layers: Sequence[int],
    target_modules: Sequence[str],
    rank: int,
    seed_offset: int,
) -> _LoRAState:
    model = getattr(adapter, "model", None)
    if model is None:
        raise ValueError("LoRA baseline requires HuggingFaceAdapter with `.model`.")

    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), Mapping) else {}
    seeds = run_cfg.get("seeds", {}) if isinstance(run_cfg.get("seeds"), Mapping) else {}
    master_seed = int(seeds.get("master", 0))

    blocks = _hf_blocks(model)
    state = _LoRAState()
    state.layers = [int(x) for x in layers]
    state.target_modules = [str(x) for x in target_modules]
    resolved: set[str] = set()

    for layer in state.layers:
        block = blocks[int(layer)]
        for mod_path in state.target_modules:
            module, resolved_path = _resolve_attr_path_any(block, mod_path)
            resolved.add(str(resolved_path))
            weight = getattr(module, "weight", None)
            if weight is None:
                raise ValueError(f"Target module has no weight: layer={layer} module={mod_path}")
            w = torch.as_tensor(weight)
            if hasattr(module, "in_features") and hasattr(module, "out_features"):
                in_features = int(getattr(module, "in_features"))
                out_features = int(getattr(module, "out_features"))
            else:
                if w.ndim != 2:
                    raise ValueError(f"Unsupported weight shape for LoRA: {tuple(w.shape)}")
                # GPT2 Conv1D uses [in, out]; treat that as canonical.
                in_features = int(w.shape[0])
                out_features = int(w.shape[1])

            # Initialize on CPU with an explicit CPU generator, then move to the module device.
            # (PyTorch requires the generator's device type to match the tensor's device.)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(
                master_seed
                + 9000
                + int(seed_offset)
                + 97 * int(layer)
                + _stable_int_seed("lora", str(layer), str(resolved_path))
            )
            A_init = (
                torch.randn(
                    (in_features, int(rank)),
                    generator=gen,
                    device="cpu",
                    dtype=torch.float32,
                )
                * 0.01
            )
            A = torch.nn.Parameter(A_init.to(device=w.device))
            B = torch.nn.Parameter(
                torch.zeros((int(rank), out_features), device=w.device, dtype=torch.float32)
            )
            state.params.extend([A, B])

            def hook_fn(_module: Any, inputs: Any, output: Any, *, A=A, B=B, state=state) -> Any:  # noqa: N803
                if not state.enabled:
                    return output
                if not inputs:
                    raise ValueError("LoRA hook missing inputs[0].")
                x = torch.as_tensor(inputs[0])
                out_t = torch.as_tensor(output)
                delta = (x @ A.to(device=x.device, dtype=x.dtype)) @ B.to(
                    device=x.device, dtype=x.dtype
                )
                if delta.shape != out_t.shape:
                    raise ValueError(
                        f"LoRA delta shape mismatch: {tuple(delta.shape)} vs {tuple(out_t.shape)}"
                    )
                return out_t + delta

            h = module.register_forward_hook(hook_fn)
            state.handles.append(h)

    state.resolved_modules = sorted(resolved)
    return state


def _eval_spec_exact_lora_hf(
    *,
    cfg: Mapping[str, Any],
    adapter: Any,
    lora: _LoRAState,
    examples: Sequence[SpecExample],
) -> Dict[str, Any]:
    if not examples:
        raise ValueError("examples must be non-empty.")

    gate_cfg = cfg.get("gate", {}) if isinstance(cfg.get("gate"), Mapping) else {}
    gate = GateSpec(wrapper_line=str(gate_cfg["wrapper_line"]), suffix=str(gate_cfg["suffix"]))
    if not all(boolqa_gate(e.prompt, gate) for e in examples):
        raise ValueError("Spec examples must be gate-true for LoRA baseline.")

    tokens = assert_or_select_answer_tokens(adapter, cfg)
    yes_id = int(tokens["yes_id"])
    no_id = int(tokens["no_id"])

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

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            end = min(len(examples), start + batch_size)
            prompts = prompts_all[start:end]
            labels = labels_all[start:end]

            toks = adapter.tokenize(prompts)
            p = answer_positions(toks["attention_mask"])
            lora.enabled = True
            try:
                logits = adapter.forward_logits(
                    input_ids=toks["input_ids"], attention_mask=toks["attention_mask"]
                )
            finally:
                lora.enabled = False
            logits_p = gather_logits_at_positions(logits, p)
            yes = logits_p[:, yes_id]
            no = logits_p[:, no_id]

            pred = (yes > no).to(dtype=torch.int64)
            failures += int((pred != labels.to(device=pred.device)).sum().item())

            correct = torch.where(labels.to(device=yes.device) == 1, yes, no)
            incorrect = torch.where(labels.to(device=yes.device) == 1, no, yes)
            margins.append((correct - incorrect).detach().cpu())

    margins_np = torch.cat(margins, dim=0).numpy()
    total = int(len(examples))
    return {
        "total": total,
        "failures": int(failures),
        "pass_rate": float(1.0 - (failures / total)),
        "min_margin": float(margins_np.min()),
        "p05_margin": float(np.quantile(margins_np, 0.05)),
    }


def _eval_collateral_lora_hf(
    *,
    cfg: Mapping[str, Any],
    adapter: Any,
    lora: _LoRAState,
    refbool_s_prompts: Sequence[str],
    refbool_l_prompts: Sequence[str],
    reftext_prompts: Sequence[str],
) -> Dict[str, Any]:
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), Mapping) else {}
    batch_size = max(1, int(eval_cfg.get("batch_size", 64)))
    resamples = int(eval_cfg.get("bootstrap_resamples", 2000))
    bootstrap_seed = int(eval_cfg.get("bootstrap_seed", 0))
    gen_cfg = (
        eval_cfg.get("generation", {}) if isinstance(eval_cfg.get("generation"), Mapping) else {}
    )
    max_new_tokens = max(0, int(gen_cfg.get("max_new_tokens", 128)))

    gate_cfg = cfg.get("gate", {}) if isinstance(cfg.get("gate"), Mapping) else {}
    gate = GateSpec(wrapper_line=str(gate_cfg["wrapper_line"]), suffix=str(gate_cfg["suffix"]))
    gate_enabled = bool(gate_cfg.get("enabled", True))

    # RefBool-S KL
    if not all(boolqa_gate(p, gate) for p in refbool_s_prompts):
        raise ValueError("RefBool-S prompts must be gate-true.")

    kls: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(refbool_s_prompts), batch_size):
            end = min(len(refbool_s_prompts), start + batch_size)
            batch = list(refbool_s_prompts[start:end])
            toks = adapter.tokenize(batch)
            p = answer_positions(toks["attention_mask"])

            lora.enabled = False
            base_logits = adapter.forward_logits(
                input_ids=toks["input_ids"], attention_mask=toks["attention_mask"]
            )
            base_p = gather_logits_at_positions(base_logits, p)

            if gate_enabled:
                lora.enabled = True
                try:
                    patch_logits = adapter.forward_logits(
                        input_ids=toks["input_ids"], attention_mask=toks["attention_mask"]
                    )
                finally:
                    lora.enabled = False
                patch_p = gather_logits_at_positions(patch_logits, p)
                kl = _kl_pq_from_logits(base_p, patch_p).detach().cpu().numpy().astype(np.float64)
            else:
                kl = np.zeros((end - start,), dtype=np.float64)
            kls.append(kl)

    kl_all = np.concatenate(kls, axis=0) if kls else np.zeros((0,), dtype=np.float64)
    s_mean = float(kl_all.mean()) if kl_all.size else 0.0
    ci_lo, ci_hi = _bootstrap_ci95(kl_all, resamples=resamples, seed=bootstrap_seed)

    # RefText KL (out-of-scope; LoRA disabled)
    if any(boolqa_gate(p, gate) for p in reftext_prompts):
        raise ValueError("RefText prompts must be gate-false.")
    t_mean = 0.0

    # RefBool-L drift via cached greedy generation; LoRA applied only on prompt forward.
    model = getattr(adapter, "model", None)
    if model is None:
        raise ValueError("LoRA baseline requires HuggingFaceAdapter with `.model`.")

    eos_id = getattr(getattr(adapter, "tokenizer", None), "eos_token_id", None)
    eos_id = int(eos_id) if eos_id is not None else None

    divergence = 0
    first_diffs: list[int] = []
    edit_norms: list[float] = []

    for p in refbool_l_prompts:
        if not boolqa_gate(p, gate):
            raise ValueError("RefBool-L prompts must be gate-true.")
        toks = adapter.tokenize([p])
        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]

        with torch.no_grad():
            lora.enabled = False
            out0 = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            logits0 = getattr(out0, "logits", None)
            past0 = getattr(out0, "past_key_values", None)
            if logits0 is None or past0 is None:
                raise ValueError("Base RefBool-L prompt forward missing logits/past_key_values.")

            if gate_enabled:
                lora.enabled = True
                out1 = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                lora.enabled = False
                logits1 = getattr(out1, "logits", None)
                past1 = getattr(out1, "past_key_values", None)
                if logits1 is None or past1 is None:
                    raise ValueError("LoRA prompt forward missing logits/past_key_values.")
            else:
                logits1 = logits0
                past1 = past0

            base_next = int(torch.as_tensor(logits0)[0, -1, :].argmax(dim=-1).item())
            patch_next = int(torch.as_tensor(logits1)[0, -1, :].argmax(dim=-1).item())

            base_tokens: list[int] = [base_next]
            patch_tokens: list[int] = [patch_next]

            base_tok = torch.tensor([[base_next]], device=input_ids.device, dtype=input_ids.dtype)
            patch_tok = torch.tensor([[patch_next]], device=input_ids.device, dtype=input_ids.dtype)

            for _ in range(max_new_tokens - 1):
                out_b = model(input_ids=base_tok, past_key_values=past0, use_cache=True)
                logits_b = getattr(out_b, "logits", None)
                past0 = getattr(out_b, "past_key_values", None)
                if logits_b is None or past0 is None:
                    raise ValueError("Base RefBool-L continuation missing logits/past_key_values.")
                base_next = int(torch.as_tensor(logits_b)[0, -1, :].argmax(dim=-1).item())
                base_tokens.append(base_next)
                base_tok = torch.tensor(
                    [[base_next]], device=input_ids.device, dtype=input_ids.dtype
                )

                # LoRA disabled after prompt forward; use base weights with patched cache.
                out_p = model(input_ids=patch_tok, past_key_values=past1, use_cache=True)
                logits_p = getattr(out_p, "logits", None)
                past1 = getattr(out_p, "past_key_values", None)
                if logits_p is None or past1 is None:
                    raise ValueError(
                        "Patched RefBool-L continuation missing logits/past_key_values."
                    )
                patch_next = int(torch.as_tensor(logits_p)[0, -1, :].argmax(dim=-1).item())
                patch_tokens.append(patch_next)
                patch_tok = torch.tensor(
                    [[patch_next]], device=input_ids.device, dtype=input_ids.dtype
                )

                if eos_id is not None and (base_next == eos_id or patch_next == eos_id):
                    break

        if base_tokens != patch_tokens:
            divergence += 1
        k = min(len(base_tokens), len(patch_tokens))
        first = next(
            (i for i in range(k) if base_tokens[i] != patch_tokens[i]),
            k if len(base_tokens) != len(patch_tokens) else max_new_tokens,
        )
        first_diffs.append(int(first))
        dist = _token_edit_distance(base_tokens, patch_tokens)
        denom = max(len(base_tokens), len(patch_tokens), 1)
        edit_norms.append(float(dist / denom))

    div_rate = float(divergence / max(1, len(refbool_l_prompts))) if refbool_l_prompts else 0.0
    mean_first = float(np.mean(np.asarray(first_diffs, dtype=np.float64))) if first_diffs else 0.0
    mean_edit = float(np.mean(np.asarray(edit_norms, dtype=np.float64))) if edit_norms else 0.0

    return {
        "refbool_s_mean_kl": float(s_mean),
        "refbool_s_ci95": (float(ci_lo), float(ci_hi)),
        "refbool_l_divergence_rate": float(div_rate),
        "refbool_l_first_diff_index": float(mean_first),
        "refbool_l_norm_edit_distance": float(mean_edit),
        "reftext_mean_kl": float(t_mean),
    }


def _train_lora_hf(
    *,
    cfg: Mapping[str, Any],
    adapter: Any,
    lora: _LoRAState,
    examples: Sequence[SpecExample],
    refbool_s_prompts: Sequence[str],
    alpha: float,
    seed_offset: int,
    resume_step: int = 0,
    on_step_end: Optional[Callable[[Sequence[torch.Tensor], Mapping[str, Any]], None]] = None,
    checkpoint_every: int = 100,
) -> Dict[str, Any]:
    if not examples:
        raise ValueError("examples must be non-empty.")
    if not refbool_s_prompts:
        raise ValueError("refbool_s_prompts must be non-empty.")
    if not lora.params:
        raise ValueError("LoRA state has no parameters.")

    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), Mapping) else {}
    seeds = run_cfg.get("seeds", {}) if isinstance(run_cfg.get("seeds"), Mapping) else {}
    master_seed = int(seeds.get("master", 0))

    optimizer_cfg = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer"), Mapping) else {}
    lr_grid = optimizer_cfg.get("lr_grid")
    lr = (
        float(lr_grid[len(lr_grid) // 2])
        if isinstance(lr_grid, list) and lr_grid
        else float(optimizer_cfg.get("lr", 3e-3))
    )
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

    objective_cfg = cfg.get("objective", {}) if isinstance(cfg.get("objective"), Mapping) else {}
    tau = float(objective_cfg.get("tau_margin", 1.0))
    beta = float(objective_cfg.get("beta_smooth", 50.0))

    tokens = assert_or_select_answer_tokens(adapter, cfg)
    yes_id = int(tokens["yes_id"])
    no_id = int(tokens["no_id"])

    perm_gen = torch.Generator(device="cpu")
    perm_gen.manual_seed(master_seed + 9100 + int(seed_offset))
    spec_perm = torch.randperm(len(examples), generator=perm_gen).tolist()
    ref_perm = torch.randperm(len(refbool_s_prompts), generator=perm_gen).tolist()
    spec_pos = (start_step * spec_batch_size) % len(spec_perm)
    ref_pos = (start_step * ref_batch_size) % len(ref_perm)

    opt = torch.optim.Adam(lora.params, lr=lr, betas=adam_betas)

    gate_cfg = cfg.get("gate", {}) if isinstance(cfg.get("gate"), Mapping) else {}
    gate_enabled = bool(gate_cfg.get("enabled", True))

    last_loss = 0.0
    for _step in range(start_step, steps):
        s_ids = [spec_perm[(spec_pos + i) % len(spec_perm)] for i in range(spec_batch_size)]
        spec_pos = (spec_pos + spec_batch_size) % len(spec_perm)
        r_ids = [ref_perm[(ref_pos + i) % len(ref_perm)] for i in range(ref_batch_size)]
        ref_pos = (ref_pos + ref_batch_size) % len(ref_perm)

        spec_batch = [examples[i] for i in s_ids]
        ref_batch = [refbool_s_prompts[i] for i in r_ids]

        opt.zero_grad(set_to_none=True)

        # Ref KL minibatch (base vs LoRA-enabled).
        toks_ref = adapter.tokenize(ref_batch)
        p_ref = answer_positions(toks_ref["attention_mask"])
        lora.enabled = False
        base_logits = adapter.forward_logits(
            input_ids=toks_ref["input_ids"], attention_mask=toks_ref["attention_mask"]
        )
        base_p = gather_logits_at_positions(base_logits, p_ref)

        if gate_enabled and float(alpha) != 0.0:
            lora.enabled = True
            try:
                patch_logits = adapter.forward_logits(
                    input_ids=toks_ref["input_ids"], attention_mask=toks_ref["attention_mask"]
                )
            finally:
                lora.enabled = False
            patch_p = gather_logits_at_positions(patch_logits, p_ref)
            kl = _kl_pq_from_logits(base_p, patch_p).mean()
        else:
            kl = torch.tensor(0.0, device=toks_ref["input_ids"].device, dtype=torch.float32)

        # Spec smooth violation proxy.
        prompts = [e.prompt for e in spec_batch]
        labels = torch.tensor(
            [int(e.label) for e in spec_batch],
            dtype=torch.int64,
            device=toks_ref["input_ids"].device,
        )
        toks_spec = adapter.tokenize(prompts)
        p_s = answer_positions(toks_spec["attention_mask"])
        lora.enabled = True
        try:
            logits_s = adapter.forward_logits(
                input_ids=toks_spec["input_ids"], attention_mask=toks_spec["attention_mask"]
            )
        finally:
            lora.enabled = False
        logits_s_p = gather_logits_at_positions(logits_s, p_s)
        yes = logits_s_p[:, yes_id]
        no = logits_s_p[:, no_id]
        correct = torch.where(labels == 1, yes, no)
        incorrect = torch.where(labels == 1, no, yes)
        margins = correct - incorrect
        v = torch.relu(torch.tensor(tau, device=margins.device) - margins)
        g_smooth = (torch.logsumexp(beta * v, dim=0) - math.log(max(1, v.numel()))) / beta

        loss = g_smooth + float(alpha) * kl
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss in LoRA training.")
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu().item())

        if on_step_end is not None and (
            _step == start_step or (_step + 1) % int(max(1, checkpoint_every)) == 0 or _step + 1 == steps
        ):
            on_step_end(
                [torch.as_tensor(p).detach() for p in lora.params],
                {
                    "step_next": int(_step + 1),
                    "steps": int(steps),
                    "alpha": float(alpha),
                    "loss": float(last_loss),
                },
            )

    return {
        "alpha": float(alpha),
        "steps": int(steps),
        "resume_step": int(start_step),
        "steps_run": int(max(0, steps - start_step)),
        "lr": float(lr),
        "loss_last": float(last_loss),
        "param_count": int(lora.parameter_count()),
        "fro_norm": float(lora.fro_norm()),
    }


def run_baselines(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
) -> Sequence[BaselineResult]:
    baselines_cfg = cfg.get("baselines", {}) if isinstance(cfg.get("baselines"), Mapping) else {}
    enabled = baselines_cfg.get("enabled", [])
    if not isinstance(enabled, list) or not all(isinstance(b, str) for b in enabled):
        enabled = []

    runtime = (
        cfg.get("_certipatch_runtime", {})
        if isinstance(cfg.get("_certipatch_runtime"), Mapping)
        else {}
    )
    resume = bool(runtime.get("resume", False))
    ckpt_root = _checkpoint_root(cfg)

    refbool_s = runtime.get("refbool_s_prompts")
    refbool_l = runtime.get("refbool_l_prompts")
    reftext = runtime.get("reftext_prompts")
    if not isinstance(refbool_s, list) or not all(isinstance(p, str) for p in refbool_s):
        refbool_s = []
    if not isinstance(refbool_l, list) or not all(isinstance(p, str) for p in refbool_l):
        refbool_l = []
    if not isinstance(reftext, list) or not all(isinstance(p, str) for p in reftext):
        reftext = []

    specs_cfg = cfg.get("specs", {}) if isinstance(cfg.get("specs"), Mapping) else {}
    specs_enabled = specs_cfg.get("enabled", [])
    if (
        not isinstance(specs_enabled, list)
        or not all(isinstance(s, str) for s in specs_enabled)
        or not specs_enabled
    ):
        specs_enabled = []

    if (not refbool_s or not refbool_l or not reftext) and specs_enabled:
        refbool_s, refbool_l, reftext = _ensure_runtime_prompt_suites(
            cfg=cfg,
            specs_enabled=[str(s) for s in specs_enabled],
            refbool_s=refbool_s,
            refbool_l=refbool_l,
            reftext=reftext,
        )

    cand_layers = _resolve_candidate_layers(cfg, adapter)
    patch_cfg = cfg.get("patch", {}) if isinstance(cfg.get("patch"), Mapping) else {}
    rank_r = int(patch_cfg.get("rank_r", 4))
    budget = _glrhp_budget(adapter, cand_layers=cand_layers, rank_r=rank_r)
    tau = float(
        (cfg.get("objective", {}) if isinstance(cfg.get("objective"), Mapping) else {}).get(
            "tau_margin", 1.0
        )
    )

    results: list[BaselineResult] = []

    def append_result(result: BaselineResult) -> None:
        results.append(result)
        if ckpt_root is not None:
            _atomic_write_json(
                _baseline_checkpoint_path(ckpt_root=ckpt_root, baseline_name=result.name),
                {
                    "name": str(result.name),
                    "artifacts_dir": str(result.artifacts_dir),
                    "metrics": dict(result.metrics),
                },
            )

    def maybe_load_result(name: str) -> Optional[BaselineResult]:
        if not resume or ckpt_root is None:
            return None
        ckpt = _baseline_checkpoint_path(ckpt_root=ckpt_root, baseline_name=name)
        if not ckpt.exists():
            return None
        loaded = _load_baseline_checkpoint(ckpt)
        if loaded is None:
            return None
        print(f"[resume] baselines: loaded {name} from {ckpt.as_posix()}")
        return loaded

    for name in enabled:
        loaded_result = maybe_load_result(name)
        if loaded_result is not None:
            results.append(loaded_result)
            continue

        if name == "base":
            patch = _make_glr_patch(cfg, cand_layers=cand_layers)
            metrics = _eval_all_specs_glr(
                cfg=cfg,
                adapter=adapter,
                patch=patch,
                refbool_s_prompts=refbool_s,
                refbool_l_prompts=refbool_l,
                reftext_prompts=reftext,
            )
            metrics["budget"] = {**budget, "trainable_params": 0, "budget_match_required": False}
            append_result(BaselineResult(name=name, artifacts_dir="", metrics=metrics))
            continue

        if not specs_enabled:
            append_result(
                BaselineResult(
                    name=name,
                    artifacts_dir="",
                    metrics={"skipped": True, "reason": "No enabled specs."},
                )
            )
            continue

        target_spec: SpecId = cast(SpecId, str(specs_enabled[0]))
        domain = _iter_domain_examples(cfg, target_spec)

        if name == "steering_vec_1l":
            if not refbool_s:
                append_result(
                    BaselineResult(
                        name=name,
                        artifacts_dir="",
                        metrics={"skipped": True, "reason": "Missing refbool_s_prompts."},
                    )
                )
                continue
            layer = int(adapter.info.n_layers) - 1
            cfg_steer = _cfg_with_explicit_candidate_layers(cfg, [layer])
            patch_like, diag = _train_steering_vec_1l(
                cfg=cfg_steer, adapter=adapter, examples=domain, layer=layer
            )
            # Cast to satisfy eval_* typing; runtime only needs params/apply_to_vectors.
            patch = cast(GLRHookPatch, patch_like)
            metrics = _eval_all_specs_glr(
                cfg=cfg_steer,
                adapter=adapter,
                patch=patch,
                refbool_s_prompts=refbool_s,
                refbool_l_prompts=refbool_l,
                reftext_prompts=reftext,
            )
            metrics["patch"] = patch_like.serialize()
            metrics["train_diagnostics"] = diag
            metrics["budget"] = {
                **budget,
                "trainable_params": int(patch_like.parameter_count()),
                "budget_match_required": False,
            }
            append_result(BaselineResult(name=name, artifacts_dir="", metrics=metrics))
            continue

        if name == "oneshot_full_alm":
            if not refbool_s:
                append_result(
                    BaselineResult(
                        name=name,
                        artifacts_dir="",
                        metrics={"skipped": True, "reason": "Missing refbool_s_prompts."},
                    )
                )
                continue
            patch = _make_glr_patch(cfg, cand_layers=cand_layers)
            solver_ckpt_path = (
                _baseline_solver_checkpoint_path(ckpt_root=ckpt_root, baseline_name=name)
                if ckpt_root is not None
                else None
            )
            solver_state: Optional[SolverState] = None
            if resume and solver_ckpt_path is not None and solver_ckpt_path.exists():
                solver_ckpt = _load_torch_checkpoint(solver_ckpt_path)
                if (
                    isinstance(solver_ckpt, Mapping)
                    and str(solver_ckpt.get("kind", "")) == "baseline_alm"
                    and str(solver_ckpt.get("baseline", "")) == str(name)
                ):
                    loaded_patch = _patch_from_payload(patch, solver_ckpt.get("patch"))
                    loaded_state = _solver_state_from_payload(solver_ckpt.get("state"))
                    if loaded_patch:
                        solver_state = loaded_state
                        inner_round_next = (
                            int(loaded_state.inner_round) if loaded_state is not None else 0
                        )
                        print(
                            "[resume] baselines:"
                            f" loaded {name} solver checkpoint from {solver_ckpt_path.as_posix()}"
                            f" (inner_round_next={inner_round_next})"
                        )

            def _on_alm_round_end(
                patch_now: GLRHookPatch,
                state_now: SolverState,
                round_meta: Mapping[str, Any],
            ) -> None:
                if solver_ckpt_path is None:
                    return
                _write_torch_checkpoint(
                    solver_ckpt_path,
                    {
                        "version": 1,
                        "kind": "baseline_alm",
                        "baseline": str(name),
                        "state": _solver_state_to_payload(state_now),
                        "patch": _patch_to_payload(patch_now),
                        "round_meta": dict(round_meta),
                    },
                )

            patch, _state, diag = solve_constrained_minimality(
                cfg=cfg,
                adapter=adapter,
                patch=patch,
                D_spec=domain,
                D_ref=refbool_s,
                state=solver_state,
                on_round_end=_on_alm_round_end,
            )
            if solver_ckpt_path is not None and solver_ckpt_path.exists():
                solver_ckpt_path.unlink(missing_ok=True)
            metrics = _eval_all_specs_glr(
                cfg=cfg,
                adapter=adapter,
                patch=patch,
                refbool_s_prompts=refbool_s,
                refbool_l_prompts=refbool_l,
                reftext_prompts=reftext,
            )
            metrics["train_diagnostics"] = diag
            metrics["budget"] = {
                **budget,
                "trainable_params": int(patch.parameter_count()),
                "budget_match_required": True,
                "within_tolerance": budget["budget_lo"]
                <= int(patch.parameter_count())
                <= budget["budget_hi"],
            }
            append_result(BaselineResult(name=name, artifacts_dir="", metrics=metrics))
            continue

        if name == "oneshot_full_mo":
            if not refbool_s:
                append_result(
                    BaselineResult(
                        name=name,
                        artifacts_dir="",
                        metrics={"skipped": True, "reason": "Missing refbool_s_prompts."},
                    )
                )
                continue
            baseline_cfg = (
                cfg.get("baseline", {}) if isinstance(cfg.get("baseline"), Mapping) else {}
            )
            alpha_grid = baseline_cfg.get("alpha_grid", [0.0, 0.01, 0.05, 0.1, 0.2])
            if not isinstance(alpha_grid, list) or not all(
                isinstance(a, (int, float)) for a in alpha_grid
            ):
                alpha_grid = [0.0, 0.01, 0.05, 0.1, 0.2]

            trials: list[Dict[str, Any]] = []
            alpha_values = [float(a) for a in alpha_grid]
            best_patch: Optional[GLRHookPatch] = None
            best_alpha: Optional[float] = None
            best_kl: Optional[float] = None

            for alpha in alpha_values:
                alpha_tag = _alpha_tag(alpha)
                trial_path = (
                    _alpha_checkpoint_path(ckpt_root=ckpt_root, baseline_name=name, alpha=alpha)
                    if ckpt_root is not None
                    else None
                )
                if resume and trial_path is not None and trial_path.exists():
                    loaded_trial = _load_trial_checkpoint(trial_path)
                    if loaded_trial is not None:
                        print(
                            "[resume] baselines:"
                            f" loaded {name} alpha={alpha} from {trial_path.as_posix()}"
                        )
                        trials.append(loaded_trial)
                        feasible = bool(loaded_trial.get("feasible", False))
                        collateral = loaded_trial.get("collateral")
                        if feasible and isinstance(collateral, Mapping):
                            kl = collateral.get("refbool_s_mean_kl")
                            if isinstance(kl, (int, float)):
                                if best_kl is None or float(kl) < float(best_kl):
                                    best_alpha = float(alpha)
                                    best_kl = float(kl)
                        continue

                solver_path = (
                    _alpha_solver_checkpoint_path(
                        ckpt_root=ckpt_root, baseline_name=name, alpha=alpha
                    )
                    if ckpt_root is not None
                    else None
                )
                patch = _make_glr_patch(cfg, cand_layers=cand_layers)
                resume_step = 0
                if resume and solver_path is not None and solver_path.exists():
                    solver_ckpt = _load_torch_checkpoint(solver_path)
                    if (
                        isinstance(solver_ckpt, Mapping)
                        and str(solver_ckpt.get("kind", "")) == "baseline_mo_alpha"
                        and str(solver_ckpt.get("baseline", "")) == str(name)
                        and str(solver_ckpt.get("alpha_tag", "")) == str(alpha_tag)
                    ):
                        loaded_patch = _patch_from_payload(patch, solver_ckpt.get("patch"))
                        if loaded_patch:
                            resume_step = int(max(0, int(solver_ckpt.get("step_next", 0))))
                            print(
                                "[resume] baselines:"
                                f" loaded {name} alpha={alpha} step={resume_step}"
                                f" from {solver_path.as_posix()}"
                            )

                def _on_mo_step_end(
                    patch_now: GLRHookPatch,
                    step_meta: Mapping[str, Any],
                ) -> None:
                    if solver_path is None:
                        return
                    _write_torch_checkpoint(
                        solver_path,
                        {
                            "version": 1,
                            "kind": "baseline_mo_alpha",
                            "baseline": str(name),
                            "alpha": float(alpha),
                            "alpha_tag": str(alpha_tag),
                            "step_next": int(max(0, int(step_meta.get("step_next", 0)))),
                            "patch": _patch_to_payload(patch_now),
                            "step_meta": dict(step_meta),
                        },
                    )

                patch, diag = solve_multiobjective(
                    cfg=cfg,
                    adapter=adapter,
                    patch=patch,
                    D_spec=domain,
                    D_ref=refbool_s,
                    alpha=alpha,
                    resume_step=int(resume_step),
                    on_step_end=_on_mo_step_end,
                )
                if solver_path is not None and solver_path.exists():
                    solver_path.unlink(missing_ok=True)
                spec_m = eval_spec_exact(cfg=cfg, adapter=adapter, patch=patch, examples=domain)
                col_m = eval_collateral(
                    cfg=cfg,
                    adapter=adapter,
                    patch=patch,
                    refbool_s_prompts=refbool_s,
                    refbool_l_prompts=refbool_l,
                    reftext_prompts=reftext,
                )
                feasible = bool(
                    int(spec_m.failures) == 0 and float(spec_m.min_margin) >= float(tau)
                )
                trials.append(
                    {
                        "alpha": float(alpha),
                        "feasible": feasible,
                        "spec": spec_m.__dict__,
                        "collateral": col_m.__dict__,
                        "train_diag": diag,
                    }
                )
                if trial_path is not None:
                    _atomic_write_json(trial_path, trials[-1])
                if feasible and (
                    best_kl is None or float(col_m.refbool_s_mean_kl) < float(best_kl)
                ):
                    best_patch = patch
                    best_alpha = float(alpha)
                    best_kl = float(col_m.refbool_s_mean_kl)

            if best_alpha is None:
                append_result(
                    BaselineResult(
                        name=name,
                        artifacts_dir="",
                        metrics={"skipped": False, "feasible": False, "trials": trials},
                    )
                )
                continue

            if best_patch is None:
                patch = _make_glr_patch(cfg, cand_layers=cand_layers)
                best_patch, _diag = solve_multiobjective(
                    cfg=cfg,
                    adapter=adapter,
                    patch=patch,
                    D_spec=domain,
                    D_ref=refbool_s,
                    alpha=float(best_alpha),
                )

            metrics = _eval_all_specs_glr(
                cfg=cfg,
                adapter=adapter,
                patch=best_patch,
                refbool_s_prompts=refbool_s,
                refbool_l_prompts=refbool_l,
                reftext_prompts=reftext,
            )
            metrics["alpha_search"] = {
                "grid": [float(a) for a in alpha_grid],
                "trials": trials,
                "selected": float(best_alpha),
            }
            metrics["budget"] = {
                **budget,
                "trainable_params": int(best_patch.parameter_count()),
                "budget_match_required": True,
                "within_tolerance": budget["budget_lo"]
                <= int(best_patch.parameter_count())
                <= budget["budget_hi"],
            }
            append_result(BaselineResult(name=name, artifacts_dir="", metrics=metrics))
            continue

        if name == "softprompt":
            info = getattr(adapter, "info", None)
            backend = str(getattr(info, "backend", ""))
            if backend != "huggingface":
                append_result(
                    BaselineResult(
                        name=name,
                        artifacts_dir="",
                        metrics={
                            "skipped": True,
                            "reason": "SoftPrompt baseline implemented for HuggingFace backend only.",
                        },
                    )
                )
                continue
            if not refbool_s:
                append_result(
                    BaselineResult(
                        name=name,
                        artifacts_dir="",
                        metrics={"skipped": True, "reason": "Missing refbool_s_prompts."},
                    )
                )
                continue

            baseline_cfg = (
                cfg.get("baseline", {}) if isinstance(cfg.get("baseline"), Mapping) else {}
            )
            alpha_grid = baseline_cfg.get("alpha_grid", [0.0, 0.01, 0.05, 0.1, 0.2])
            if not isinstance(alpha_grid, list) or not all(
                isinstance(a, (int, float)) for a in alpha_grid
            ):
                alpha_grid = [0.0, 0.01, 0.05, 0.1, 0.2]
            k = int(baseline_cfg.get("k_virtual_tokens", 32))

            softprompt_trials: list[Dict[str, Any]] = []
            best_softprompt: Optional[Dict[str, Any]] = None
            checkpoint_every = _checkpoint_step_interval(cfg, default=100)

            alpha_values = [float(a) for a in alpha_grid]
            for idx, alpha in enumerate(alpha_values):
                alpha_tag = _alpha_tag(alpha)
                trial_path = (
                    _alpha_checkpoint_path(ckpt_root=ckpt_root, baseline_name=name, alpha=alpha)
                    if ckpt_root is not None
                    else None
                )
                if resume and trial_path is not None and trial_path.exists():
                    loaded_trial = _load_trial_checkpoint(trial_path)
                    if loaded_trial is not None:
                        print(
                            "[resume] baselines:"
                            f" loaded {name} alpha={alpha} from {trial_path.as_posix()}"
                        )
                        softprompt_trials.append(loaded_trial)
                        loaded_feasible = bool(loaded_trial.get("feasible", False))
                        loaded_collateral = loaded_trial.get("collateral_metrics")
                        if loaded_feasible and isinstance(loaded_collateral, Mapping):
                            kl = loaded_collateral.get("refbool_s_mean_kl")
                            if isinstance(kl, (int, float)):
                                if best_softprompt is None or float(kl) < float(
                                    best_softprompt["collateral_metrics"]["refbool_s_mean_kl"]
                                ):
                                    best_softprompt = loaded_trial
                        continue

                solver_path = (
                    _alpha_solver_checkpoint_path(
                        ckpt_root=ckpt_root, baseline_name=name, alpha=alpha
                    )
                    if ckpt_root is not None
                    else None
                )
                resume_step = 0
                init_softprompt: Optional[torch.Tensor] = None
                if resume and solver_path is not None and solver_path.exists():
                    solver_ckpt = _load_torch_checkpoint(solver_path)
                    if (
                        isinstance(solver_ckpt, Mapping)
                        and str(solver_ckpt.get("kind", "")) == "baseline_softprompt_alpha"
                        and str(solver_ckpt.get("baseline", "")) == str(name)
                        and str(solver_ckpt.get("alpha_tag", "")) == str(alpha_tag)
                    ):
                        soft_raw = solver_ckpt.get("softprompt")
                        if soft_raw is not None:
                            init_softprompt = torch.as_tensor(soft_raw, dtype=torch.float32)
                            resume_step = int(max(0, int(solver_ckpt.get("step_next", 0))))
                            print(
                                "[resume] baselines:"
                                f" loaded {name} alpha={alpha} step={resume_step}"
                                f" from {solver_path.as_posix()}"
                            )

                def _on_softprompt_step(
                    soft_now: torch.Tensor,
                    step_meta: Mapping[str, Any],
                ) -> None:
                    if solver_path is None:
                        return
                    _write_torch_checkpoint(
                        solver_path,
                        {
                            "version": 1,
                            "kind": "baseline_softprompt_alpha",
                            "baseline": str(name),
                            "alpha": float(alpha),
                            "alpha_tag": str(alpha_tag),
                            "step_next": int(max(0, int(step_meta.get("step_next", 0)))),
                            "softprompt": torch.as_tensor(soft_now).detach().to(
                                device="cpu", dtype=torch.float32
                            ),
                            "step_meta": dict(step_meta),
                        },
                    )

                soft, train_diag = _train_softprompt_hf(
                    cfg=cfg,
                    adapter=adapter,
                    examples=domain,
                    refbool_s_prompts=refbool_s,
                    k_virtual_tokens=k,
                    alpha=float(alpha),
                    seed_offset=int(idx),
                    init_softprompt=init_softprompt,
                    resume_step=int(resume_step),
                    on_step_end=_on_softprompt_step,
                    checkpoint_every=int(checkpoint_every),
                )
                if solver_path is not None and solver_path.exists():
                    solver_path.unlink(missing_ok=True)

                softprompt_spec_metrics: dict[str, Any] = {}
                for sid in specs_enabled:
                    ex = _iter_domain_examples(cfg, cast(SpecId, sid))
                    softprompt_spec_metrics[str(sid)] = _eval_spec_exact_softprompt_hf(
                        cfg=cfg, adapter=adapter, softprompt=soft, examples=ex
                    )

                col_metrics = _eval_collateral_softprompt_hf(
                    cfg=cfg,
                    adapter=adapter,
                    softprompt=soft,
                    refbool_s_prompts=refbool_s,
                    refbool_l_prompts=refbool_l,
                    reftext_prompts=reftext,
                )

                target = softprompt_spec_metrics[str(target_spec)]
                feasible = bool(
                    int(target.get("failures", 0)) == 0
                    and float(target.get("min_margin", 0.0)) >= float(tau)
                )

                trial = {
                    "alpha": float(alpha),
                    "feasible": bool(feasible),
                    "spec_metrics": softprompt_spec_metrics,
                    "collateral_metrics": col_metrics,
                    "train_diagnostics": train_diag,
                    "patch": {
                        "family": "SoftPrompt",
                        "k_virtual_tokens": int(k),
                        "parameter_count": int(soft.numel()),
                        "fro_norm": float(
                            torch.as_tensor(soft, dtype=torch.float32).pow(2).sum().sqrt().item()
                        ),
                    },
                }
                softprompt_trials.append(trial)
                if trial_path is not None:
                    _atomic_write_json(trial_path, trial)

                if feasible and (
                    best_softprompt is None
                    or float(col_metrics["refbool_s_mean_kl"])
                    < float(best_softprompt["collateral_metrics"]["refbool_s_mean_kl"])
                ):
                    best_softprompt = trial

            if best_softprompt is None:
                append_result(
                    BaselineResult(
                        name=name,
                        artifacts_dir="",
                        metrics={"skipped": False, "feasible": False, "trials": softprompt_trials},
                    )
                )
                continue

            trainable_params = int(best_softprompt["patch"]["parameter_count"])
            metrics = {
                "spec_metrics": best_softprompt["spec_metrics"],
                "collateral_metrics": best_softprompt["collateral_metrics"],
                "patch": best_softprompt["patch"],
                "alpha_search": {
                    "grid": [float(a) for a in alpha_grid],
                    "trials": softprompt_trials,
                    "selected": float(best_softprompt["alpha"]),
                },
                "budget": {
                    **budget,
                    "trainable_params": trainable_params,
                    "budget_match_required": True,
                    "within_tolerance": budget["budget_lo"]
                    <= trainable_params
                    <= budget["budget_hi"],
                },
            }
            append_result(BaselineResult(name=name, artifacts_dir="", metrics=metrics))
            continue

        if name == "lora":
            info = getattr(adapter, "info", None)
            backend = str(getattr(info, "backend", ""))
            if backend != "huggingface":
                append_result(
                    BaselineResult(
                        name=name,
                        artifacts_dir="",
                        metrics={
                            "skipped": True,
                            "reason": "LoRA baseline implemented for HuggingFace backend only.",
                        },
                    )
                )
                continue
            if not refbool_s:
                append_result(
                    BaselineResult(
                        name=name,
                        artifacts_dir="",
                        metrics={"skipped": True, "reason": "Missing refbool_s_prompts."},
                    )
                )
                continue

            baseline_cfg = (
                cfg.get("baseline", {}) if isinstance(cfg.get("baseline"), Mapping) else {}
            )
            alpha_grid = baseline_cfg.get("alpha_grid", [0.0, 0.01, 0.05, 0.1, 0.2])
            if not isinstance(alpha_grid, list) or not all(
                isinstance(a, (int, float)) for a in alpha_grid
            ):
                alpha_grid = [0.0, 0.01, 0.05, 0.1, 0.2]

            rank = int(baseline_cfg.get("lora_rank", 4))
            target_modules = baseline_cfg.get("lora_target_modules", ["attn.c_proj"])
            if not isinstance(target_modules, list) or not all(
                isinstance(m, str) for m in target_modules
            ):
                target_modules = ["attn.c_proj"]

            layers_spec = baseline_cfg.get("lora_layers", "candidate_layers")
            if layers_spec == "candidate_layers":
                layers = [int(x) for x in cand_layers]
            elif isinstance(layers_spec, list) and all(isinstance(x, int) for x in layers_spec):
                layers = [int(x) for x in layers_spec]
            else:
                layers = [int(x) for x in cand_layers]

            lora_trials: list[Dict[str, Any]] = []
            best_lora: Optional[Dict[str, Any]] = None
            checkpoint_every = _checkpoint_step_interval(cfg, default=100)

            alpha_values = [float(a) for a in alpha_grid]
            for idx, alpha in enumerate(alpha_values):
                alpha_tag = _alpha_tag(alpha)
                trial_path = (
                    _alpha_checkpoint_path(ckpt_root=ckpt_root, baseline_name=name, alpha=alpha)
                    if ckpt_root is not None
                    else None
                )
                if resume and trial_path is not None and trial_path.exists():
                    loaded_trial = _load_trial_checkpoint(trial_path)
                    if loaded_trial is not None:
                        print(
                            "[resume] baselines:"
                            f" loaded {name} alpha={alpha} from {trial_path.as_posix()}"
                        )
                        lora_trials.append(loaded_trial)
                        loaded_feasible = bool(loaded_trial.get("feasible", False))
                        loaded_collateral = loaded_trial.get("collateral_metrics")
                        if loaded_feasible and isinstance(loaded_collateral, Mapping):
                            kl = loaded_collateral.get("refbool_s_mean_kl")
                            if isinstance(kl, (int, float)):
                                if best_lora is None or float(kl) < float(
                                    best_lora["collateral_metrics"]["refbool_s_mean_kl"]
                                ):
                                    best_lora = loaded_trial
                        continue

                solver_path = (
                    _alpha_solver_checkpoint_path(
                        ckpt_root=ckpt_root, baseline_name=name, alpha=alpha
                    )
                    if ckpt_root is not None
                    else None
                )
                lora_state = _install_lora_hf(
                    cfg=cfg,
                    adapter=adapter,
                    layers=layers,
                    target_modules=target_modules,
                    rank=rank,
                    seed_offset=int(idx),
                )
                try:
                    resume_step = 0
                    if resume and solver_path is not None and solver_path.exists():
                        solver_ckpt = _load_torch_checkpoint(solver_path)
                        if (
                            isinstance(solver_ckpt, Mapping)
                            and str(solver_ckpt.get("kind", "")) == "baseline_lora_alpha"
                            and str(solver_ckpt.get("baseline", "")) == str(name)
                            and str(solver_ckpt.get("alpha_tag", "")) == str(alpha_tag)
                        ):
                            params_raw = solver_ckpt.get("lora_params")
                            if (
                                isinstance(params_raw, list)
                                and len(params_raw) == len(lora_state.params)
                            ):
                                for p, saved in zip(lora_state.params, params_raw):
                                    p.data.copy_(
                                        torch.as_tensor(saved, device=p.device, dtype=p.dtype)
                                    )
                                resume_step = int(max(0, int(solver_ckpt.get("step_next", 0))))
                                print(
                                    "[resume] baselines:"
                                    f" loaded {name} alpha={alpha} step={resume_step}"
                                    f" from {solver_path.as_posix()}"
                                )

                    def _on_lora_step(
                        lora_params: Sequence[torch.Tensor],
                        step_meta: Mapping[str, Any],
                    ) -> None:
                        if solver_path is None:
                            return
                        _write_torch_checkpoint(
                            solver_path,
                            {
                                "version": 1,
                                "kind": "baseline_lora_alpha",
                                "baseline": str(name),
                                "alpha": float(alpha),
                                "alpha_tag": str(alpha_tag),
                                "step_next": int(max(0, int(step_meta.get("step_next", 0)))),
                                "lora_params": [
                                    torch.as_tensor(p).detach().to(device="cpu", dtype=torch.float32)
                                    for p in lora_params
                                ],
                                "step_meta": dict(step_meta),
                            },
                        )

                    train_diag = _train_lora_hf(
                        cfg=cfg,
                        adapter=adapter,
                        lora=lora_state,
                        examples=domain,
                        refbool_s_prompts=refbool_s,
                        alpha=float(alpha),
                        seed_offset=int(idx),
                        resume_step=int(resume_step),
                        on_step_end=_on_lora_step,
                        checkpoint_every=int(checkpoint_every),
                    )
                    if solver_path is not None and solver_path.exists():
                        solver_path.unlink(missing_ok=True)

                    lora_spec_metrics: dict[str, Any] = {}
                    for sid in specs_enabled:
                        ex = _iter_domain_examples(cfg, cast(SpecId, sid))
                        lora_spec_metrics[str(sid)] = _eval_spec_exact_lora_hf(
                            cfg=cfg, adapter=adapter, lora=lora_state, examples=ex
                        )

                    col_metrics = _eval_collateral_lora_hf(
                        cfg=cfg,
                        adapter=adapter,
                        lora=lora_state,
                        refbool_s_prompts=refbool_s,
                        refbool_l_prompts=refbool_l,
                        reftext_prompts=reftext,
                    )

                    target = lora_spec_metrics[str(target_spec)]
                    feasible = bool(
                        int(target.get("failures", 0)) == 0
                        and float(target.get("min_margin", 0.0)) >= float(tau)
                    )

                    patch_info = {
                        "family": "LoRA",
                        "rank": int(rank),
                        "layers": [int(x) for x in layers],
                        "target_modules": list(target_modules),
                        "target_modules_resolved": [str(x) for x in lora_state.resolved_modules],
                        "parameter_count": int(lora_state.parameter_count()),
                        "fro_norm": float(lora_state.fro_norm()),
                    }
                    trial = {
                        "alpha": float(alpha),
                        "feasible": bool(feasible),
                        "spec_metrics": lora_spec_metrics,
                        "collateral_metrics": col_metrics,
                        "train_diagnostics": train_diag,
                        "patch": patch_info,
                    }
                    lora_trials.append(trial)
                    if trial_path is not None:
                        _atomic_write_json(trial_path, trial)

                    if feasible and (
                        best_lora is None
                        or float(col_metrics["refbool_s_mean_kl"])
                        < float(best_lora["collateral_metrics"]["refbool_s_mean_kl"])
                    ):
                        best_lora = trial
                finally:
                    lora_state.remove()

            if best_lora is None:
                append_result(
                    BaselineResult(
                        name=name,
                        artifacts_dir="",
                        metrics={"skipped": False, "feasible": False, "trials": lora_trials},
                    )
                )
                continue

            trainable_params = int(best_lora["patch"]["parameter_count"])
            metrics = {
                "spec_metrics": best_lora["spec_metrics"],
                "collateral_metrics": best_lora["collateral_metrics"],
                "patch": best_lora["patch"],
                "alpha_search": {
                    "grid": [float(a) for a in alpha_grid],
                    "trials": lora_trials,
                    "selected": float(best_lora["alpha"]),
                },
                "budget": {
                    **budget,
                    "trainable_params": trainable_params,
                    "budget_match_required": True,
                    "within_tolerance": budget["budget_lo"]
                    <= trainable_params
                    <= budget["budget_hi"],
                },
            }
            append_result(BaselineResult(name=name, artifacts_dir="", metrics=metrics))
            continue

        append_result(
            BaselineResult(
                name=name,
                artifacts_dir="",
                metrics={"skipped": True, "reason": f"Unknown baseline: {name}"},
            )
        )

    return results
