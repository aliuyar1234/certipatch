# Decision Log (Append-only)

Rules:
- Never edit past entries; only append.
- Any design change MUST be logged here and in SSOT.
- Include rationale and impact on comparability.

## Template
### YYYY-MM-DD — Decision Title
- Change:
- Rationale:
- Impact:
- Migration steps:
- Approver:

## Entries
### 2026-02-02 — Initial SSOT lock
- Change: Established CertiPatch SSOT v1 (name, specs, patch family GLR-HP, ALM+CEGIS protocol, baselines, figures, certificates).
- Rationale: Prevent drift; enable Codex end-to-end build without questions.
- Impact: All artifacts and certificates conform to schema_version 1.0.
- Migration steps: N/A.
- Approver: au / brain-doc

### 2026-02-02 — Audit patch v1.1 (padding-safe answer position, LoRA budget fix, schemas, wordlists, generation semantics)
- Change:
  - Clarified answer position as `p = attention_mask.sum(dim=1) - 1` (per example), not `seq_len-1` (padding-safe).
  - Fixed LoRA budget-matched baseline to use rank-4 on `attn.c_proj` in the 4 candidate layers (exact ~24,576 params for GPT‑2 small).
  - Standardized hookpoint naming to `resid_post` everywhere (incl. certificate example).
  - Added license-free wordlists under `assets/wordlists/` for collateral suite generation + synthetic RefText.
  - Added machine-checkable JSON schemas under `schemas/` (certificate/manifest/run_record/metrics).
  - Clarified RefBool‑L generation semantics: patch applied only during initial prompt forward pass; disabled for subsequent cached generation steps.
- Rationale: Remove implementation ambiguity that could cause silent bugs or unfair backend differences; reduce Codex guesswork.
- Impact:
  - No semantic change if you already used left-padding and `p=seq_len-1`; otherwise fixes incorrect indexing.
  - LoRA baseline results will change because rank is corrected to match budget.
  - New schemas/wordlists are additive; certificates remain schema_version 1.0.
- Migration steps:
  - Update code to compute `p` from attention masks and to implement RefBool‑L generation semantics.
  - Re-run LoRA baselines; re-emit certificates (new hashes).
- Approver: au / brain-doc

### 2026-02-06 - Compute-optimized full-tier default (single seed, no separate scaling model)
- Change:
  - Set default full-tier seed list to `[0]` in `configs/paper_full.yaml`.
  - Disable separate scaling runs by default by setting `paper.models.scaling == paper.models.main`.
  - Align SSOT/docs and run status expectations with this default.
- Rationale:
  - Reduce end-to-end GPU time while preserving the core paper evidence (main model runs, baselines, ablations, compositionality, and additional specs).
  - Minimize risk of long-run interruption loss by shortening wall-clock to first complete paper-quality artifact set.
- Impact:
  - Default `--tier full` matrix is materially smaller than the prior 3-seed + 1B scaling profile.
  - Multi-seed/scaling evidence remains possible by explicit config override.
- Migration steps:
  - Resume existing `paper_full` runs with `--resume` under the updated config.
  - If extra budget becomes available, re-enable scaling by setting `paper.models.scaling` to a different model and extending `paper.seeds`.
- Approver: user (runtime decision)
