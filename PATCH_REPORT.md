# PATCH_REPORT.md — Cumulative Patch Log (Newest First)

## v4 → v5 (SSOT consistency + YAML safety patch)

### Why this patch exists
This patch removes two high-risk “Codex drift” failure modes:
1) YAML parsing ambiguity: unquoted keys `yes`/`no` are parsed as booleans by many YAML 1.1 parsers (including PyYAML), breaking config schema validation and downstream logic.
2) Baseline budget-matching semantics were inconsistent across SSOT docs, which would cause incorrect implementation choices and reviewer-facing confusion.

### What changed (file-by-file)
- `configs/default.yaml`
  - Quoted the keys `"yes"` and `"no"` under `answer_tokens.primary` and `answer_tokens.fallback` to prevent YAML boolean-key coercion.
  - Result: `default.yaml` now validates against `schemas/config_schema.json` under PyYAML, fail-closed.
- `EXPERIMENTS.md`
  - Clarified baseline categories:
    - Trainable PEFT baselines MUST be budget-matched (±10%).
    - Diagnostic baselines (`base`, `steering_vec_1l`) MAY be under-budget but MUST report param counts.
- `07_Baselines_Ablations.md`
  - Rewrote the budget-matching section to be model-agnostic and consistent with EXPERIMENTS.md.
- `00_SSOT.md`
  - Updated the baseline summary to match the clarified budget-matching rule (trainable PEFT baselines only; diagnostic baselines exempt).
- `certipatch/eval/baselines.py`
  - Updated the docstring to match the clarified budget-matching rule.
- `MANIFEST.sha256`
  - Regenerated to reflect the patched file contents (self-hash rule preserved).

---

# PATCH_REPORT.md — certipatch_handoff_v2 → certipatch_handoff_v3

This patch fills SSOT packaging gaps while keeping the project flexible via configs.

## Summary of changes (high level)
- Added a minimal Python repo scaffold (pyproject + package skeleton) with exhaustive docstrings and pseudocode.
- Added normative SSOT docs required for Codex to implement end-to-end without external context.
- Added config schema and normalized YAML configs to a single surface.
- Made LaTeX paper skeleton compile-ready and aligned it with fixed figure/table filenames.
- Added a global repository integrity manifest with a documented self-hash rule.
- Removed all scaffold-marker language and any deferred-language; uncertainty is expressed as FAIL-CLOSED.

## File-by-file change list

### Added — repository scaffold
- `pyproject.toml`
  - Defines minimal packaging + optional deps for torch/HF/TransformerLens/viz.
- `certipatch/__init__.py`
- `certipatch/config.py`
  - Config loading/validation contract (schema-driven, fail-closed).
- `certipatch/models/__init__.py`
- `certipatch/models/load_model.py`
  - Adapter contract + fail-closed hookpoint discovery semantics (backend-agnostic).
- `certipatch/hooks.py`
  - Gate + answer-position rules + hook application contract.
- `certipatch/patch_families.py`
  - GLR-HP patch family scaffold (state layout + serialization contract).
- `certipatch/specs/__init__.py`
- `certipatch/specs/compare_2d.py`
- `certipatch/specs/parity_4d.py`
- `certipatch/specs/balance_paren_14.py`
- `certipatch/specs/compare_6d_strat.py`
  - Coverage-bounded spec scaffold with deterministic strata rules.
- `certipatch/cegis/__init__.py`
- `certipatch/cegis/loop.py`
- `certipatch/cegis/trainer.py`
- `certipatch/cegis/counterexamples.py`
  - CEGIS + ALM schedules and counterexample discovery contracts.
- `certipatch/eval/__init__.py`
- `certipatch/eval/metrics.py`
- `certipatch/eval/baselines.py`
  - Metric definitions and baseline contracts.
- `certipatch/artifacts/__init__.py`
- `certipatch/artifacts/certificate.py`
- `certipatch/artifacts/report.py`
- `certipatch/artifacts/verifier.py`
  - Replayable certificate + verifier contracts, including manifest self-hash policy.
- `scripts/reproduce_paper.py`
  - One-command reproduction driver scaffold and required outputs contract.

### Added — global integrity manifest
- `MANIFEST.sha256`
  - Covers every file in the repository, including itself via the documented zeroed-self-hash rule.

### Added — required SSOT docs
- `SPEC.md`
- `ALGORITHMS.md`
- `EXPERIMENTS.md`
- `FIGURES_TABLES.md`
- `CLAIMS_TO_EVIDENCE.md`
- `DATA_INVENTORY.md`
- `DATA_GENERATION.md`
- `REPRODUCE.md`
- `DECISIONS.md`
- `BLOCKERS.md` (set to `NONE`)

### Added / Updated — config surface + schema
- `schemas/config_schema.json` (new)
  - Declares required config keys: model backend+id, answer tokens, hookpoints, rank/layers, enabled specs, coverage plan params, seeds.
- `schemas/README.md` (updated)
  - Documents config schema and clarifies that repository integrity is via MANIFEST.sha256.
- `configs/default.yaml` (rewritten)
  - Normalized to required keys; revision may be null and is recorded by the adapter in run_record/certificates.
- Updated overlays:
  - `configs/compare2d_certipatch.yaml`
  - `configs/parity4d_certipatch.yaml`
  - `configs/balance14_certipatch.yaml`
  - `configs/compare6d_strat_certipatch.yaml`
  - `configs/compositionality.yaml`
  - `configs/lora_baseline.yaml`
  - `configs/softprompt_baseline.yaml`
  - `configs/oneshot_full_alm.yaml`
  - `configs/oneshot_full_mo.yaml`
- Added:
  - `configs/paper_full.yaml` (enables full paper suite without hardcoding a specific model)

### Updated — LaTeX paper skeleton (compile-ready)
- `paper/latex/main.tex` (rewritten)
  - Removed external conference style dependency and inserted a complete abstract.
- `paper/latex/sections/*.tex` (rewritten)
  - Minimal text; includes fixed figure/table filenames from FIGURES_TABLES.md.
- `paper/latex/references.bib` (updated)
  - Empty but valid bib file (no external lookups).
- `paper/latex/figures/*.pdf` (added)
  - Stub PDFs for all required figures (artifact slots; overwritten by reproduction script).
- `paper/latex/tables/tab01_main_results.tex` (added)
- `paper/latex/tables/tab02_ablations.tex` (added)
- `paper/latex/figures/README.md` and `paper/latex/tables/README.md` (updated)
  - List expected filenames.

### Updated — existing docs for consistency and no-defer-language
- `00_SSOT.md`
  - Removed deferred phrasing in rank ablation line.
- `10_Paper_Writing_LaTeX_Plan.md`
  - Describes compile-ready skeleton without ambiguous marker language.
- `11_Progress_Tracker.md`
  - Standardized progress state labels to NOT_STARTED/IN_PROGRESS/DONE/BLOCKED.
- `STATUS.yaml`
  - Standardized task state labels to NOT_STARTED.
- `18_Appendix_Math_and_Design_Rationale.md`
  - Rephrased long-horizon collateral explanation without deferred wording.
- `paper/FIGURE_SPECS.md`
  - Added a note that FIGURES_TABLES.md is authoritative for filenames.

## Compatibility note
This v3 handoff intentionally contains scaffolds only: signatures + exhaustive docstrings + pseudocode.
Codex is expected to implement the full system using these contracts and SSOT docs, without relying on external context.

## v4 patch — SSOT consistency + prompt canonicalization

This patch fixes SSOT gaps that would cause silent drift for an implementer who has no prior context.

### Changes
- `README.md`
  - Declared SSOT precedence order and clarified that `SPEC.md` is the top authority.
  - Added a concrete quick-start and removed any ambiguity about which docs win conflicts.

- `00_SSOT.md`
  - Added an explicit banner: if conflicts arise, `SPEC.md`/`ALGORITHMS.md`/`EXPERIMENTS.md` win.

- `02_Specs_Domains.md`
  - Made parity formatting explicit: `n_str = str(n)` (no zero padding). Removed ambiguous wording.

- `DATA_GENERATION.md`
  - Added canonical, hash-critical question templates for all specs (compare_2d, parity_4d, balance_paren_14, compare_6d_strat).

- `certipatch/specs/compare_2d.py`
  - Removed the redundant substring "Answer Yes or No." from the question template.

- `certipatch/specs/parity_4d.py`
  - Removed zero padding and removed the redundant substring "Answer Yes or No." from the question template.
  - Updated module docstring to match the canonical (hash-critical) formatting.

- `certipatch/specs/balance_paren_14.py`
  - Canonicalized quoting to double quotes around the parentheses string and removed the redundant substring.
  - Updated module docstring to match the canonical (hash-critical) quoting.

- `certipatch/specs/compare_6d_strat.py`
  - Removed the redundant substring from the question template.

- `04_Optimization_ALM_CEGIS.md`
  - Removed the "fixed τ" phrasing and made τ explicitly configurable via `objective.tau_margin`.

- `MANIFEST.sha256`
  - Regenerated to match all file changes, preserving the documented self-hash rule.
