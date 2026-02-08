"""certipatch.cegis.counterexamples

Counterexample discovery is deterministic and scope-aware.

Two regimes:
1) Enumerable domains:
   - exact sweep over full domain
   - return all failures (or the K hardest failures by margin)

2) Coverage-bounded domains:
   - evaluate certified coverage set (strata)
   - run additional bounded search budgets:
        - fixed-seed interior samples
        - local perturbations around lowest-margin points
   - return any discovered failures with metadata indicating where found

Fail-closed:
- For non-enumerable domains, counterexample absence means:
    "no counterexamples were found within the certified coverage and search budgets"
  It MUST NOT be interpreted as global correctness.

The counterexample set emitted into artifacts MUST be hashed and stored as JSONL.

This file is a scaffold. Implementations are omitted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Sequence

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
from certipatch.specs import SpecExample, SpecId


@dataclass(frozen=True)
class Counterexample:
    prompt: str
    label: int
    pred: int
    margin: float
    meta: Dict[str, str]


def _make_patch_fn(patch: GLRHookPatch, *, layer: int) -> Callable[[Any], Any]:
    def _fn(h: Any) -> Any:
        return patch.apply_to_vectors(h, layer=layer)

    return _fn


def find_counterexamples(
    *,
    cfg: Mapping[str, Any],
    adapter: ModelAdapter,
    patch: GLRHookPatch,
    spec_id: SpecId,
) -> List[Counterexample]:
    """Return a list of counterexamples discovered under the current patch."""
    progress_on = progress_enabled(cfg)
    progress_cfg = progress_config(cfg)
    log_every_batches = max(1, int(progress_cfg.get("log_every_batches", 25)))
    run_id = ""
    run_cfg = cfg.get("run", {})
    if isinstance(run_cfg, Mapping):
        run_id = str(run_cfg.get("run_id", "")).strip()
    run_dir_p = run_dir(cfg) if progress_on else None
    cex_log = (run_dir_p / "cex_progress.jsonl") if run_dir_p is not None else None
    t0 = time.monotonic()

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
        missing_layers = [layer for layer in cand_layers if layer not in patch.params]
        if missing_layers:
            raise ValueError(f"Patch missing parameters for candidate layers: {missing_layers}")

    batch_size = (
        int(cfg.get("evaluation", {}).get("batch_size", 64))
        if isinstance(cfg.get("evaluation"), Mapping)
        else 64
    )
    batch_size = max(1, batch_size)

    objective_cfg = cfg.get("objective", {}) if isinstance(cfg.get("objective"), Mapping) else {}
    tau = float(objective_cfg.get("tau_margin", 1.0))

    def iter_examples() -> Iterator[SpecExample]:
        if spec_id == "compare_2d":
            from certipatch.specs.compare_2d import iter_domain

            yield from iter_domain(cfg)
            return
        if spec_id == "parity_4d":
            from certipatch.specs.parity_4d import iter_domain

            yield from iter_domain(cfg)
            return
        if spec_id == "balance_paren_14":
            from certipatch.specs.balance_paren_14 import iter_domain

            yield from iter_domain(cfg)
            return
        if spec_id == "compare_6d_strat":
            from certipatch.specs.compare_6d_strat import (
                iter_additional_search_samples,
                iter_certified_coverage,
            )

            yield from iter_certified_coverage(cfg)
            yield from iter_additional_search_samples(cfg)
            return

        raise ValueError(f"Unknown spec_id: {spec_id}")

    out: list[Counterexample] = []
    batch: list[SpecExample] = []
    batches_done = 0
    examples_seen = 0

    if cex_log is not None:
        append_jsonl(
            cex_log,
            {
                "timestamp_utc": utc_now_iso(),
                "event": "cex_start",
                "run_id": run_id,
                "spec_id": str(spec_id),
                "batch_size": int(batch_size),
                "tau_margin": float(tau),
            },
        )
        if run_id:
            print(f"[cex] {run_id} spec={spec_id} start batch_size={batch_size}", flush=True)

    def flush() -> None:
        nonlocal out, batch, batches_done, examples_seen
        if not batch:
            return
        n_batch = int(len(batch))

        prompts = [e.prompt for e in batch]
        labels = torch.tensor([int(e.label) for e in batch], dtype=torch.int64)

        gate_pred = [boolqa_gate(p, gate) for p in prompts]
        if not all(gate_pred):
            bad = gate_pred.index(False)
            raise ValueError(
                f"Spec example out-of-scope (gate=false) in counterexample search: index {bad}"
            )

        toks = adapter.tokenize(prompts)
        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]
        p_idx = answer_positions(attention_mask)

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
                            layer=layer,
                            batch_positions=p_idx,
                            gate_mask=gate_mask,
                            patch_fn=_make_patch_fn(patch, layer=int(layer)),
                        )
                    )
            with torch.no_grad():
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
        pred = (yes_logits > no_logits).to(dtype=torch.int64)

        correct_logits = torch.where(
            labels.to(device=yes_logits.device) == 1, yes_logits, no_logits
        )
        incorrect_logits = torch.where(
            labels.to(device=yes_logits.device) == 1, no_logits, yes_logits
        )
        margins = (correct_logits - incorrect_logits).detach().cpu().tolist()
        preds = pred.detach().cpu().tolist()

        for e, pr, m in zip(batch, preds, margins):
            if float(m) >= float(tau):
                continue  # counterexamples are margin violations (not only misclassifications)
            meta = dict(e.meta)
            meta["spec_id"] = str(spec_id)
            out.append(
                Counterexample(
                    prompt=e.prompt,
                    label=int(e.label),
                    pred=int(pr),
                    margin=float(m),
                    meta=meta,
                )
            )

        batch = []
        batches_done += 1
        examples_seen += n_batch

        if cex_log is not None and (batches_done == 1 or batches_done % log_every_batches == 0):
            append_jsonl(
                cex_log,
                {
                    "timestamp_utc": utc_now_iso(),
                    "event": "cex_progress",
                    "run_id": run_id,
                    "spec_id": str(spec_id),
                    "batches_done": int(batches_done),
                    "examples_seen": int(examples_seen),
                    "counterexamples_found": int(len(out)),
                    "elapsed_s": float(time.monotonic() - t0),
                },
            )
            if run_id:
                print(
                    f"[cex] {run_id} spec={spec_id} seen={examples_seen} cex={len(out)}",
                    flush=True,
                )

    for ex in iter_examples():
        batch.append(ex)
        if len(batch) >= batch_size:
            flush()
    flush()

    if cex_log is not None:
        append_jsonl(
            cex_log,
            {
                "timestamp_utc": utc_now_iso(),
                "event": "cex_end",
                "run_id": run_id,
                "spec_id": str(spec_id),
                "batches_done": int(batches_done),
                "examples_seen": int(examples_seen),
                "counterexamples_found": int(len(out)),
                "elapsed_s": float(time.monotonic() - t0),
            },
        )
        if run_id:
            print(
                f"[cex] {run_id} spec={spec_id} end seen={examples_seen} cex={len(out)}",
                flush=True,
            )

    return out


def select_hardest(counterexamples: Sequence[Counterexample], k_add: int) -> List[Counterexample]:
    """Select K counterexamples with lowest margin (hardest) deterministically.

    Tie-break (MUST):
      - sort by (margin ascending, prompt lexicographic) and take first k_add.
    """
    k = int(k_add)
    if k <= 0:
        return []
    return sorted(counterexamples, key=lambda c: (c.margin, c.prompt))[:k]


def select_random(
    counterexamples: Sequence[Counterexample], k_add: int, *, seed: int
) -> List[Counterexample]:
    """Select K counterexamples uniformly at random, deterministically.

    Determinism rule:
      - First sort by prompt to get a canonical ordering.
      - Then apply a fixed-seed permutation and take the first k.
    """
    k = int(k_add)
    if k <= 0:
        return []
    c_sorted = sorted(counterexamples, key=lambda c: c.prompt)
    if k >= len(c_sorted):
        return list(c_sorted)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(len(c_sorted), generator=gen).tolist()
    return [c_sorted[i] for i in perm[:k]]
