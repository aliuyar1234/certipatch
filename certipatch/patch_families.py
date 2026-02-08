"""certipatch.patch_families

Patch families define parameterizations of inference-time hookpoint modifications.

v1 family (SSOT): GLR-HP (Gated Low-Rank Residual Hook Patch)

Definition:
  For each layer l in candidate layer set L:
    h[l, p] <- h[l, p] + s(x) * U_l (V_l^T h[l, p])

Where:
  - h[l, p] is the residual stream at layer l and answer position p
  - U_l in R^{d x r}, V_l in R^{d x r}
  - s(x) is a deterministic gate predicate shared across all specs

Effective layers:
  A layer is "effective" if ||U_l||_F^2 + ||V_l||_F^2 >= threshold.

This file is a scaffold:
- Provide class signatures, state layout, serialization contract, and pseudocode.
- Do not provide full training/eval implementation here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class GLRHPConfig:
    """Configuration for the GLR-HP patch family."""

    rank_r: int
    candidate_layers: List[int]
    effective_layer_threshold: float


class GLRHookPatch:
    """Gated Low-Rank Residual Hook Patch (GLR-HP).

    State:
      - For each candidate layer l:
          U_l: [d_model, r]
          V_l: [d_model, r]

    Required methods (Codex MUST implement):
      - `init_parameters(d_model, seed)`
      - `forward_vector(h_vec, layer_index)` returning patched vector
      - `serialize()` -> bytes or dict
      - `load(serialized)`
      - `effective_layers()` -> List[int]
      - `parameter_count()` -> int
      - `fro_norm()` -> float
    """

    def __init__(self, cfg: GLRHPConfig):
        self.cfg = cfg
        # Parameters are intentionally not allocated in this scaffold.
        self.params: Dict[int, Dict[str, Any]] = {}

    def init_parameters(self, d_model: int, seed: int) -> None:
        """Initialize parameters deterministically.

        Pseudocode:
            set_torch_seed(seed)
            for l in candidate_layers:
                U_l = normal(0, 0.01, [d_model, r])
                V_l = normal(0, 0.01, [d_model, r])
                store

        Fail-closed:
            - Ensure rank_r > 0 and d_model > 0.
        """
        if self.cfg.rank_r <= 0:
            raise ValueError("rank_r must be > 0")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if not self.cfg.candidate_layers:
            raise ValueError("candidate_layers must be non-empty")

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        self.params = {}
        for layer in self.cfg.candidate_layers:
            U = torch.randn(d_model, self.cfg.rank_r, generator=gen, dtype=torch.float32) * 0.01
            V = torch.randn(d_model, self.cfg.rank_r, generator=gen, dtype=torch.float32) * 0.01
            self.params[int(layer)] = {"U": U, "V": V}

    def apply_to_vectors(self, h: Any, layer: int) -> Any:
        """Apply the patch transform to a batch of vectors.

        Input:
            h: [N, d_model] tensor of vectors extracted at the patched positions.
            layer: layer index within candidate_layers.

        Output:
            h_patched: [N, d_model]

        Pseudocode:
            (U, V) = params[layer]
            z = h @ V          # [N, r]
            delta = z @ U.T    # [N, d]
            return h + delta
        """
        if layer not in self.params:
            raise ValueError(f"Layer {layer} not initialized for this patch.")

        h_t = torch.as_tensor(h)
        if h_t.ndim != 2:
            raise ValueError(f"h must have shape [N, d_model], got {tuple(h_t.shape)}")

        U = torch.as_tensor(self.params[layer]["U"], device=h_t.device, dtype=h_t.dtype)
        V = torch.as_tensor(self.params[layer]["V"], device=h_t.device, dtype=h_t.dtype)

        if U.ndim != 2 or V.ndim != 2:
            raise ValueError("Patch parameters must be rank-2 matrices.")
        if U.shape != V.shape:
            raise ValueError("U and V must have the same shape.")
        if h_t.shape[1] != U.shape[0]:
            raise ValueError("Dimension mismatch between h and patch parameters.")

        z = h_t @ V  # [N, r]
        delta = z @ U.T  # [N, d]
        return h_t + delta

    def delta_vectors(self, h: Any, layer: int) -> Any:
        """Return only the additive delta (without adding it to h)."""
        if layer not in self.params:
            raise ValueError(f"Layer {layer} not initialized for this patch.")

        h_t = torch.as_tensor(h)
        if h_t.ndim != 2:
            raise ValueError(f"h must have shape [N, d_model], got {tuple(h_t.shape)}")

        U = torch.as_tensor(self.params[layer]["U"], device=h_t.device, dtype=h_t.dtype)
        V = torch.as_tensor(self.params[layer]["V"], device=h_t.device, dtype=h_t.dtype)

        if U.ndim != 2 or V.ndim != 2:
            raise ValueError("Patch parameters must be rank-2 matrices.")
        if U.shape != V.shape:
            raise ValueError("U and V must have the same shape.")
        if h_t.shape[1] != U.shape[0]:
            raise ValueError("Dimension mismatch between h and patch parameters.")

        z = h_t @ V  # [N, r]
        return z @ U.T  # [N, d]

    def __add__(self, other: object) -> "GLRHookPatch":
        """Add two GLR-HP patches additively at hookpoints by rank concatenation.

        If patches have ranks r1 and r2, the sum is represented as a single GLR-HP patch with rank (r1+r2)
        by concatenating the (U,V) factors along the rank dimension. This corresponds to applying both
        deltas computed from the same base activation and adding them (no cross-terms).
        """
        if not isinstance(other, GLRHookPatch):
            return NotImplemented
        if not self.params:
            return other
        if not other.params:
            return self

        if list(self.cfg.candidate_layers) != list(other.cfg.candidate_layers):
            raise ValueError("Cannot add patches with different candidate_layers.")
        if float(self.cfg.effective_layer_threshold) != float(other.cfg.effective_layer_threshold):
            raise ValueError("Cannot add patches with different effective_layer_threshold.")

        rank = int(self.cfg.rank_r) + int(other.cfg.rank_r)
        out = GLRHookPatch(
            cfg=GLRHPConfig(
                rank_r=rank,
                candidate_layers=[int(x) for x in self.cfg.candidate_layers],
                effective_layer_threshold=float(self.cfg.effective_layer_threshold),
            )
        )

        for layer in out.cfg.candidate_layers:
            if layer not in self.params or layer not in other.params:
                raise ValueError("Both patches must define parameters for all candidate layers.")
            U1 = torch.as_tensor(self.params[layer]["U"])
            V1 = torch.as_tensor(self.params[layer]["V"])
            U2 = torch.as_tensor(other.params[layer]["U"], device=U1.device, dtype=U1.dtype)
            V2 = torch.as_tensor(other.params[layer]["V"], device=V1.device, dtype=V1.dtype)

            if U1.shape[0] != U2.shape[0] or V1.shape[0] != V2.shape[0]:
                raise ValueError("Cannot add patches with different d_model.")
            out.params[layer] = {
                "U": torch.cat([U1, U2], dim=1),
                "V": torch.cat([V1, V2], dim=1),
            }

        return out

    def effective_layers(self) -> List[int]:
        """Return layers with norm above threshold."""
        if not self.params:
            return []

        threshold = float(self.cfg.effective_layer_threshold)
        effective: list[int] = []
        for layer in self.cfg.candidate_layers:
            if layer not in self.params:
                continue
            U = torch.as_tensor(self.params[layer]["U"])
            V = torch.as_tensor(self.params[layer]["V"])
            norm2 = float(U.pow(2).sum().item() + V.pow(2).sum().item())
            if norm2 >= threshold:
                effective.append(int(layer))
        return sorted(set(effective))

    def parameter_count(self) -> int:
        """Return total parameter count."""
        if not self.params:
            return 0
        total = 0
        for layer in self.cfg.candidate_layers:
            if layer not in self.params:
                continue
            U = torch.as_tensor(self.params[layer]["U"])
            V = torch.as_tensor(self.params[layer]["V"])
            total += int(U.numel() + V.numel())
        return total

    def fro_norm(self) -> float:
        """Return Frobenius norm of all parameters."""
        if not self.params:
            return 0.0
        total_sq = 0.0
        for layer in self.cfg.candidate_layers:
            if layer not in self.params:
                continue
            U = torch.as_tensor(self.params[layer]["U"])
            V = torch.as_tensor(self.params[layer]["V"])
            total_sq += float(U.pow(2).sum().item() + V.pow(2).sum().item())
        return float(total_sq**0.5)

    def serialize(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict for certificate/report.

        MUST include:
          - family name
          - rank
          - candidate layers
          - parameter tensors (stored in .pt in actual implementation)
          - fro_norm and effective_layers

        """
        per_layer_norm2: Dict[str, float] = {}
        for layer in self.cfg.candidate_layers:
            if layer not in self.params:
                continue
            U = torch.as_tensor(self.params[layer]["U"])
            V = torch.as_tensor(self.params[layer]["V"])
            per_layer_norm2[str(layer)] = float(U.pow(2).sum().item() + V.pow(2).sum().item())

        return {
            "family": "GLR-HP",
            "rank_r": int(self.cfg.rank_r),
            "candidate_layers": [int(x) for x in self.cfg.candidate_layers],
            "effective_layer_threshold": float(self.cfg.effective_layer_threshold),
            "parameter_count": self.parameter_count(),
            "fro_norm": self.fro_norm(),
            "effective_layers": self.effective_layers(),
            "per_layer_norm2": per_layer_norm2,
        }
