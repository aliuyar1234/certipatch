# Determinism (CUDA/CPU)

CertiPatch aims to be *fail‑closed and replayable*: a certificate should verify by re-running the exact evaluation
from `run_record.json` and reproducing `metrics.json` within the configured tolerance.

## What the code does

The project centralizes determinism setup in `certipatch/determinism.py`:

- `set_global_determinism(cfg)` (called by `scripts/reproduce_paper.py` and by the verifier):
  - Sets seeds: Python `random`, NumPy, Torch (CPU and CUDA).
  - Enables deterministic algorithms when supported (`torch.use_deterministic_algorithms(True)`).
  - Sets CuDNN determinism (`torch.backends.cudnn.deterministic=True`, `benchmark=False`).
  - Disables TF32 (`torch.backends.cuda.matmul.allow_tf32=False`, `torch.backends.cudnn.allow_tf32=False`).
  - Best-effort sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` (required by CUDA for some deterministic GEMMs).

Hardware metadata is captured in `run_record.json` under `environment.hardware` (best-effort).

## Practical guidance

- Prefer `run.dtype: float32` for paper runs.
- Avoid changing batch size/sequence padding behavior between training and verification.
- If CUDA determinism is unavailable for an op on your system, the verifier should fail (metric mismatch) rather than
  silently accept drift.

## Quick check

Run the same command twice with the same config and seeds and compare the resulting `runs/<run_id>/metrics.json`.
For example:

```bash
python scripts/reproduce_paper.py --config configs/compare2d_certipatch.yaml --tier toy
python scripts/reproduce_paper.py --config configs/compare2d_certipatch.yaml --tier toy
```
