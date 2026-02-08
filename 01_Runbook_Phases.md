# Runbook: Phases, Tasks, Gates, and DONE Rules

## Purpose
This is the operational runbook Codex MUST follow. It is written so an engineer with zero context can execute the project without drift or guesswork.

## Global constraints
- Determinism is mandatory. Every run MUST be reproducible from hashes + seeds.
- “Fail‑closed” is mandatory: if coverage is incomplete or hashes mismatch, the verifier MUST output FAIL.
- No changes to wrapper, gate, or token choices without updating the SSOT and a Decision Log entry.
- Every run MUST produce a certificate and a run manifest.

---

## Phase map (high level)
Phase 0 — Environment & determinism
Phase 1 — Data generators + hashing
Phase 2 — Patch + hookpoints
Phase 3 — Metrics (spec, collateral, complexity)
Phase 4 — Constrained optimizer (ALM) + feasibility checking
Phase 5 — CertiPatch outer loop (CEGIS)
Phase 6 — Baselines + ablations
Phase 7 — Experiment execution order (gates)
Phase 8 — Figures/tables and paper writing

Each phase below has:
- **Deliverables** (files and artifacts)
- **Verification checks** (what must be true before moving on)
- **Failure modes** (what typically goes wrong)
- **Done marker** instructions

---

## Progress tracking system (mandatory)
### How to track progress without searching the entire codebase
You SHALL maintain these files:
1) `STATUS.yaml` — the single progress dashboard.
2) `session_logs/YYYY-MM-DD_sessionN.md` — a brief session log template.
3) `artifacts/DONE.md` — checklist of produced paper artifacts.

Rules:
- Every task in `STATUS.yaml` has a unique ID (e.g., P2.T3).
- When finished, mark as `DONE` and record the commit hash (or zip hash), and the path to the artifact.
- If a task is blocked, mark as `BLOCKED` with a reason.
- NEVER delete tasks from STATUS.yaml; only transition state.

---

## Phase 0 — Environment & determinism
### Tasks
P0.T1 Create pinned environment spec
- Choose python version and pin dependencies.
- Must pin: torch, transformers, accelerate, numpy, scipy, pandas, matplotlib.
- Output: `env/requirements.lock.txt` or `env/uv.lock` (choose one).

P0.T2 Determinism flags
- Implement a utility that sets:
  - python hash seed
  - numpy seed
  - torch seed (CPU and CUDA)
  - `torch.use_deterministic_algorithms(True)` if supported
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
- Output: `docs/determinism.md` describing exact flags and environment variables.

P0.T3 Hardware logging
- Record GPU name, driver versions, CUDA version in `run_record.json` automatically.

### Verification
- Run a tiny test twice and ensure identical logits for a fixed prompt and seed.

### Done marker
- Update `STATUS.yaml` for P0.* and include the hash of the environment lock file.

---

## Phase 1 — Data generators + hashing
### Specs generators (must implement exactly)
See `02_Specs_Domains.md` for exact enumerations and canonical ordering.

### Tasks
P1.T1 Implement canonical prompt builder
- Takes `question_text` and wraps it in the fixed wrapper.
- Must normalize trailing whitespace when checking gate, but MUST NOT change emitted wrapper.

P1.T2 Implement enumerators
- COMPARE-2D: lexicographic enumeration of a then b.
- PARITY-4D: increasing n from 0..9999.
- BALANCE-PAREN-14: increasing length, then lexicographic over bitstring mapping.
- Output: each enumerator yields `(id_string, prompt_text, label_bool)`.

P1.T3 Implement coverage-bounded generator for COMPARE-6D-STRAT
- Must implement the strata definitions and deterministic sampling.
- Output must be a *named set* with per-stratum counts.

P1.T4 Implement reference suite generators
- RefBool-S: deterministic generator producing BoolQA wrapper prompts NOT overlapping with spec prompts.
- RefBool-L: deterministic generator producing wrapper prompts asking for explanation.

P1.T5 Hashing rules
- Domain hash for enumerable domains: sha256 of concatenated canonical prompts separated by newline + label bits.
- Coverage plan hash: sha256 of JSON of strata definitions + seeds + counts.
- Suite hash: sha256 of concatenated prompts (and any labels if present).

### Verification
- Hashes must be stable across runs/machines.
- No duplicates within enumerated domain.

### Done marker
- Save generated prompt files (optional) OR guarantee regeneration determinism; in either case record hashes.
- Mark P1 tasks DONE in STATUS.yaml.

---

## Phase 2 — Patch + hookpoints
### Tasks
P2.T1 Implement gate function s(x)
- Must implement the exact predicate from SSOT.
- Must include unit tests for gate positives/negatives.

P2.T2 Implement GLR-HP patch module
- Must support:
  - rank r=4
  - candidate layers list
  - additive composition of multiple patches (for A+B, A→B)
- Must expose:
  - parameter count
  - Frobenius norm
  - effective layer detection threshold (e.g., layer active if sqrt(||U||^2+||V||^2) > 1e-3)

P2.T3 Hook integration
Two supported backends:
A) **TransformerLens** (preferred)
- Hookpoint: `blocks.{l}.hook_resid_post` at the **answer positions p(x)** (last non-pad token per example).
- Ensure patch applies to the residual stream after MLP+attn block.

B) **HuggingFace hooks** (fallback)
- Register forward hook on the appropriate module output tensor.
- Must ensure shape alignment and correct position index.

### Verification
- Apply patch with φ=0 => identical outputs to base.
- Apply patch with random φ but gate=0 => identical outputs.
- Apply patch with random φ and gate=1 => outputs change.

---

## Phase 3 — Metrics
### Spec metrics
- failure count (exact for enumerable)
- pass rate per stratum (for coverage)
- margin statistics: min and p05
- Ensure logits correspond to next token after "Answer:".
  - Critical: compute per-example answer positions `p = attention_mask.sum(dim=1)-1` and index logits with `logits[range(B), p]`.

### Collateral metrics
RefBool-S:
- KL(base || patched) at answer position, full vocab.
- ΔNLL on base argmax token.

RefBool-L:
- Generation MUST apply the patch only during the initial prompt forward pass (to patch the answer position) and MUST disable the patch for subsequent cached generation steps. See FAQ Q14.
- divergence rate in greedy generation up to 128 tokens
- first-diff index (token-level)
- normalized edit distance (token-level Levenshtein on generated continuations; MUST match Figure spec)

### Complexity metrics
- parameter count
- Fro norm
- effective layers count
- runtime overhead per forward

### Verification
- Unit tests on toy logits for KL correctness.
- Margin computed correctly for correct token.

---

## Phase 4 — Constrained optimizer (ALM)
### Tasks
P4.T1 Implement smooth max violation for training
- Use `g_smooth = logsumexp(beta*relu)/beta`, beta=50.
- Also compute `g_true` as max relu over a full evaluation set periodically.

P4.T2 Implement ALM schedule
- inner loop: Adam 2000 steps
- update lambda and mu exactly as SSOT
- stopping: must satisfy g_true==0 on active set and no improvement in collateral for 200 steps.

P4.T3 Hyperparameter grid
- Must run deterministic grid; choose best feasible, then minimal KL.

### Verification
- On a small toy domain (e.g., COMPARE with a,b in 0..9), can reach 0 failures.

---

## Phase 5 — CertiPatch outer loop (CEGIS)
### Tasks
P5.T1 Active set maintenance
- initial sample n0 (recommend 512 for 10k domains; 2048 for 32k)
- store active set as prompt IDs (not raw strings) for efficiency, but include raw prompts in artifacts.

P5.T2 Counterexample search
- Enumerable: exact sweep each outer iter; gather violations and margins.
- Non-enumerable: evaluate certified coverage set each iter; if violations exist, add hardest.

P5.T3 Outer termination
- stop when no counterexamples in certified scope.
- Always record outer iteration logs and counterexample history.

### Verification
- Outer loop reduces failures monotonically on the active set (feasibility).
- On enumerable domains, should converge in ≤10 outer iterations.

---

## Phase 6 — Baselines + ablations
### Baselines
Implement exactly as SSOT and ensure param budget matching.
- SteeringVec: best effort budget match; also include steering with matched total params (4 layers x vectors).
- SoftPrompt: k=32 tokens for GPT2; for other models choose k = round(params/d_model).
- LoRA: pick a small set of projection matrices and rank so param count matches; log exact count.

### Ablations
- No minimality: multiobjective with fixed α
- No CEGIS: train on fixed initial set only
- No collateral: α=0
- No gate: s(x)=1
- rank=1
- single layer

### Verification
- Each baseline must be able to hit 0 failures on COMPARE-2D and PARITY-4D if possible (tune within budget).
- If LoRA cannot, report as negative result, but ensure training setup is correct.

---

## Phase 7 — Experiment execution order (fail-fast gates)
You SHALL execute in this order:

Gate 0: tokenization check (Yes/No or fallback).
Gate 1: toy domain (COMPARE range 00..19) — ensure feasibility achieved quickly.
Gate 2: COMPARE-2D full domain exact closure.
Gate 3: PARITY-4D full domain exact closure.
Gate 4: collateral sweep + minimality curves on both.
Gate 5: baselines on COMPARE-2D and PARITY-4D to 0 failures (if possible).
Gate 6: compositionality experiment (A-only, B-only, A+B, A→B, B→A, Joint AB).
Gate 7: BALANCE-PAREN-14 (may fail under strict collateral; still must run).
Gate 8: COMPARE-6D-STRAT coverage-bounded certificate.

---

## Phase 8 — Figures/tables + paper writing
Use `05_Evaluation_and_Figures.md` and `paper/latex/` templates.
- All figures must be auto-generated from artifacts.
- Tables must be exported as CSV and included in LaTeX via `\input{}` or `\csvautotabular`.

---

## Definition of DONE
A phase is DONE only if:
- All tasks are marked DONE in STATUS.yaml.
- There is at least one certificate.json produced by the pipeline for a toy run.
- Verifier passes on the toy certificate.