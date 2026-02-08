"""certipatch.hooks

Hook utilities shared across training, evaluation, and verification.

Core rules (MUST):
- The patch is applied only when the gate fires (scope control).
- The patch is applied at the answer position p(x), computed per-example:
      p = attention_mask.sum(dim=1) - 1
- The patch modifies only the selected token positions and must preserve all other tokens.

This file is a scaffold. Implementations are intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class GateSpec:
    """Defines the gating predicate.

    The default gate is a BoolQA wrapper predicate:
      - prompt contains cfg['gate']['wrapper_line'] exactly on some line
      - prompt ends with cfg['gate']['suffix'] on the last line (after stripping trailing whitespace)

    Gate MUST be deterministic and MUST be identical across specs.
    """

    wrapper_line: str
    suffix: str


def boolqa_gate(prompt: str, gate: GateSpec) -> bool:
    """Return True iff the prompt is in-scope for patch application.

    Fail-closed:
    - If the prompt contains non-UTF8 characters or cannot be normalized, return False.
    - Gate is strict string match; no regex; no fuzzy matching.
    """
    try:
        prompt.encode("utf-8")
    except UnicodeEncodeError:
        return False

    lines = prompt.splitlines()
    if gate.wrapper_line not in lines:
        return False

    trimmed = prompt.rstrip()
    if not trimmed.endswith(gate.suffix):
        return False
    last_line = trimmed.splitlines()[-1] if trimmed else ""
    return last_line == gate.suffix


def answer_positions(attention_mask: Any) -> Any:
    """Compute answer position indices p for each example in a batch.

    p = sum(attention_mask) - 1, per example.

    Fail-closed:
    - If any example has sum(attention_mask)==0, abort (invalid batch).
    - p must be within [0, T-1] for each example.

    Returns:
        p indices shaped [B] (same device as attention_mask).
    """
    mask = torch.as_tensor(attention_mask)
    if mask.ndim != 2:
        raise ValueError(f"attention_mask must be rank-2 [B,T], got shape {tuple(mask.shape)}")
    if mask.shape[1] <= 0:
        raise ValueError("attention_mask must have T>0")

    sums = mask.to(dtype=torch.int64).sum(dim=1)
    if torch.any(sums <= 0):
        raise ValueError("Invalid batch: at least one example has attention_mask sum == 0.")

    p = sums - 1
    T = mask.shape[1]
    if torch.any((p < 0) | (p >= T)):
        raise ValueError("Computed answer position out of bounds.")
    return p


def gather_logits_at_positions(logits: Any, p: Any) -> Any:
    """Gather logits at positions p.

    logits: [B, T, V]
    p: [B]

    Return:
        logits_p: [B, V]
    """
    logits_t = torch.as_tensor(logits)
    p_t = torch.as_tensor(p, dtype=torch.int64, device=logits_t.device)

    if logits_t.ndim != 3:
        raise ValueError(f"logits must have shape [B,T,V], got {tuple(logits_t.shape)}")
    if p_t.ndim != 1:
        raise ValueError(f"p must have shape [B], got {tuple(p_t.shape)}")
    B, T, _ = logits_t.shape
    if p_t.shape[0] != B:
        raise ValueError("Batch size mismatch between logits and p.")
    if torch.any((p_t < 0) | (p_t >= T)):
        raise ValueError("Position indices out of bounds for logits.")

    batch_idx = torch.arange(B, device=logits_t.device)
    return logits_t[batch_idx, p_t, :]


@dataclass
class HookApplication:
    """Represents a registered hook and its runtime configuration."""

    kind: str
    layer: int
    handle: Any


def apply_hookpoint_patch(
    adapter: Any,
    *,
    kind: str,
    layer: int,
    batch_positions: Any,
    gate_mask: Any,
    patch_fn: Callable[[Any], Any],
) -> HookApplication:
    """Register a hook that applies patch_fn at selected positions.

    Inputs:
      - batch_positions: [B] int tensor p(x)
      - gate_mask: [B] bool tensor; only examples with gate_mask True are patched.
      - patch_fn: maps activation vectors [N, d] -> [N, d] or returns delta.

    Hook semantics (MUST):
      - For each example i where gate_mask[i] is True, apply patch_fn to activation[i, p[i], :].
      - All other tokens are unchanged.

    Fail-closed:
      - If adapter cannot register the requested hook, abort.
      - If activation shapes do not match expected (B, T, d), abort.

    Returns:
      A HookApplication with a removable handle.
    """
    pos_t = torch.as_tensor(batch_positions, dtype=torch.int64)
    gate_t = torch.as_tensor(gate_mask, dtype=torch.bool)

    if pos_t.ndim != 1:
        raise ValueError("batch_positions must be rank-1 [B].")
    if gate_t.ndim != 1:
        raise ValueError("gate_mask must be rank-1 [B].")
    if pos_t.shape[0] != gate_t.shape[0]:
        raise ValueError("batch_positions and gate_mask must have the same batch size.")

    returns_delta = bool(getattr(patch_fn, "__certipatch_returns_delta__", False))

    def hook_fn(activation: Any, _batch_idx: Any, _pos_idx: Any) -> Any:
        act = torch.as_tensor(activation)
        if act.ndim != 3:
            raise ValueError(f"Expected activation shape [B,T,d], got {tuple(act.shape)}")
        B, T, d_model = act.shape
        if pos_t.shape[0] != B:
            raise ValueError("Activation batch size does not match batch_positions.")

        pos = pos_t.to(device=act.device)
        gate = gate_t.to(device=act.device)

        if torch.any((pos < 0) | (pos >= T)):
            raise ValueError("batch_positions out of bounds for activation tensor.")

        idx = torch.nonzero(gate, as_tuple=False).flatten()
        if idx.numel() == 0:
            return act

        vecs = act[idx, pos[idx], :]
        if vecs.ndim != 2 or vecs.shape[1] != d_model:
            raise ValueError("Failed to gather activation vectors for patching.")

        out_vecs = patch_fn(vecs)
        out_t = torch.as_tensor(out_vecs, device=act.device)
        if out_t.shape != vecs.shape:
            raise ValueError("patch_fn must return a tensor shaped [N, d_model].")
        if returns_delta:
            out_t = vecs + out_t

        out_act = act.clone()
        out_act[idx, pos[idx], :] = out_t
        return out_act

    handle = adapter.register_hook(kind=kind, layer=layer, hook_fn=hook_fn)
    return HookApplication(kind=kind, layer=layer, handle=handle)
