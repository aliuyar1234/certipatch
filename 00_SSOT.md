# CertiPatch SSOT (Single Source of Truth)

> **Important**: `SPEC.md` is normative. If `00_SSOT.md` conflicts with `SPEC.md`/`ALGORITHMS.md`/`EXPERIMENTS.md`, Codex MUST follow those.
> `00_SSOT.md` is a quickstart index and is kept for convenience.


> **Purpose**: This folder is the **Single Source of Truth (SSOT)** for Codex to build the complete CertiPatch project end‑to‑end (code, runs, data, figures, and LaTeX paper) **without asking questions**.
>
> **Non-negotiables**: deterministic, fail‑closed, coverage-bounded certificates, decisive ablations that neutralize reviewer attacks A/B/C.

## TL;DR (do this in order)
1. Implement model loader + tokenization checks (Yes/No single-token fallback).
2. Implement GLR‑HP patch (gated low-rank hookpoint patch) at resid_post (post-block residual), answer position.
3. Implement constrained minimality optimizer (Augmented Lagrangian schedule) + evaluation (exact failure counts).
4. Implement CertiPatch outer loop (CEGIS-style counterexample-guided constraint set growth).
5. Implement data generators for specs + collateral suites (RefBool-S and RefBool-L) with hashing.
6. Implement baselines (SteeringVec, SoftPrompt, LoRA, OneShot full-domain).
7. Run experiments in the runbook order; emit artifacts and certificates.
8. Render figures/tables exactly as specified; write paper from LaTeX skeleton.

---

## 0. Glossary / Notation (frozen)
- **LM**: frozen causal language model `f_θ` with tokenizer.
- **Prompt**: input string `x` (already includes wrapper + question + "Answer:").
- **Gate** `s(x) ∈ {0,1}`: indicates prompt is in the certified BoolQA wrapper family.
- **Answer tokens**: two single tokens `t_yes`, `t_no`. Primary: `" Yes"`/`" No"`. Fallback: `" true"`/`" false"`.
- **Prediction**: argmax between logits of `t_yes` and `t_no` at the next-token position after `"Answer:"`.
- **Answer position** `p(x)`: index of the last non-padding token in the tokenized prompt. MUST be computed as `p = attention_mask.sum(dim=1) - 1` (per example). This avoids padding bugs and avoids searching for the "Answer:" token.
- **Spec**: programmatic labeler `Spec(x)` over domain `X_spec` defined by generator parameters.
- **Patch** `φ`: inference-time hookpoint patch (GLR‑HP) added to residual stream at selected layers/positions.
- **Collateral**: divergence between base and patched distributions on a gated reference set; primary metric is mean KL.
- **Certificate**: replayable empirical artifact with hashes, scope, coverage, counterexamples, and metrics; **NOT** a formal proof.

---

## 1. Executive Decisions (locked)
### 1.1 Project name
**CertiPatch** (must appear everywhere; do not use “SpecPatch”).

### 1.2 Models
- **Paper full default (compute-optimized)**: EleutherAI/pythia-410m-deduped via HuggingFace.
- **Dev/ablation model**: openai-community/gpt2 via HuggingFace.
- **Optional scaling extension**: Pythia-1B (EleutherAI) via HuggingFace only when extra compute is available.
- Must record:
  - model ID, exact revision/commit (or exact version tag), tokenizer ID.
  - torch + transformers versions
  - deterministic flags and seeds.

### 1.3 Prompt wrapper (gate-firing family)
All certified prompts MUST be exactly:

```
Instruction: Answer with a single token: Yes or No.
Question: {QUESTION_TEXT}
Answer:
```

Gate predicate `s(x)=1` iff:
1) The prompt contains the exact substring `Instruction: Answer with a single token: Yes or No.` and
2) The prompt ends with the exact suffix `Answer:` (optionally followed by a single trailing space; normalize by stripping right-side whitespace before checking).

**Important**: the gate is shared across all specs and collateral suites. This is deliberate to make collateral and patch compositionality nontrivial.

### 1.4 Patch family v1 (GLR‑HP)
- Hookpoint: **resid_post (post-block residual stream)** at **answer position**.
- **Batching rule (critical)**: prompts MAY be padded. The implementation MUST compute per-example answer positions `p(x)` from `attention_mask` and apply the patch only at those indices. Do NOT assume `p = seq_len-1` unless you enforce left padding everywhere.
- Candidate layers: 4 fixed layers: `{⌊n/4⌋, ⌊n/2⌋, ⌊3n/4⌋, n-1}` where `n` is #blocks.
- Patch per layer ℓ:

**Core operator**:
Let `h` be the residual vector at `(layer ℓ, position p=answer index)`.
Apply:
`h ← h + s(x) * U_ℓ @ (V_ℓ^T @ h)`.

Shapes:
- `h`: `[d_model]`
- `V_ℓ`: `[d_model, r]`
- `V_ℓ^T @ h`: `[r]`
- `U_ℓ`: `[d_model, r]`
- `U_ℓ @ (...)`: `[d_model]`
Rank `r=4` (v1). Ablation uses `r=1`.

**Complexity controls**:
- L2 penalty on U,V
- Group lasso across layers to encourage ≤2 effective layers (but parameters are still present; group lasso may push them near zero).

### 1.5 Specs/domains (final list)
All use the shared wrapper and are labeled programmatically.

**Spec A: COMPARE‑2D (enumerable, 10k)**
- Parameters: `a,b ∈ {00..99}`
- Label: Yes iff `a > b`
- Domain size: 10,000 (exact sweep every outer iter)

**Spec B: PARITY‑4D (enumerable, 10k)**
- Parameter: `n ∈ {0..9999}`
- Label: Yes iff `n` even
- Domain size: 10,000 (exact sweep)

**Spec C: BALANCE‑PAREN‑14 (enumerable, 32,767)**
- Parameter: string `s ∈ {(,)}^{≤14}`
- Label: balanced parentheses via stack
- Domain size: 32,767 (exact sweep)

**Spec D: COMPARE‑6D‑STRAT (non-enumerable; coverage-bounded)**
- Parameters: `a,b ∈ {000000..999999}`; true domain size 10^12
- Certified scope: a fixed stratified coverage set of 80,000 examples + bounded search budgets (see `02_Specs_Domains.md`).

### 1.6 Collateral suites (final)
Collateral MUST be measured where gate=1.

- **RefBool‑S** (short, distributional): 20,000 BoolQA wrapper prompts not in any spec domains.
  - Primary metric: mean KL(base || patched) at the answer position full vocab.
  - Secondary: ΔNLL on base argmax token.

- **RefBool‑L** (stronger long-form drift): 1,000 BoolQA wrapper prompts requesting Yes/No + one-sentence explanation.
  - Greedy decoding, max_new_tokens=128
  - Patch semantics: patch MUST be applied only during the initial prompt forward pass (prompt caching); disable patch for subsequent cached generation steps.
  - Metrics: divergence rate, first diff index, normalized edit distance.

- **RefText** (gate=0 sanity): 5,000 natural text prompts; expected near-zero drift; report for scoping clarity.

### 1.7 Baselines (required)

Baselines are required for fairness and for reviewer hardening.

Budget matching rule:
- Trainable PEFT baselines (SoftPrompt, LoRA, OneShot-FullDomain-MO, OneShot-FullDomain-ALM) MUST be within ±10% of GLR-HP trainable parameter count for the chosen model.
- Diagnostic baselines are exempt (but MUST report parameter counts):
  - Base (0 parameters)
  - SteeringVec-1L (intentionally small “single direction” steering sanity check)

Compute GLR-HP trainable parameter count for the chosen model:
- Let `d = d_model`, `r = rank_r`, `L = |candidate_layers|`.
- `P_GLRHP = 2 * d * r * L`.

Example (non-normative): if `d=768`, `r=4`, `L=4` then `P_GLRHP=24,576`.

Baselines:
1) Base (no patch)
2) SteeringVec-1L (diagnostic; under-budget) — additive vector at one layer
3) OneShot-FullDomain-MO (train on full enumerable domain once; multiobjective)
4) OneShot-FullDomain-ALM (train on full enumerable domain once; ALM without constraint generation)
5) SoftPrompt (budget-matched; virtual token prefix)
6) LoRA (budget-matched; applied to fixed layer subset as specified in 07_Baselines_Ablations.md)
### 1.8 Success criteria (must-pass)
On COMPARE‑2D and PARITY‑4D:
- CertiPatch MUST reach **0 failures** and **0 margin violations** (min_margin ≥ τ) on the full domain.
- CertiPatch MUST strictly improve **minimality-at-feasibility** vs the two one-shot baselines:
  - RefBool‑S mean KL must be ≥20% lower than **OneShot‑FullDomain‑ALM** *or* patch ||φ||_F must be ≥20% lower (report both).
  - Additionally, CertiPatch MUST not be dominated by OneShot‑FullDomain‑MO on the (KL, ||φ||_F) Pareto plane.
- CertiPatch SHOULD use ≤2 effective layers at the chosen Pareto-knee (group lasso working), but this is a soft goal; the hard goal is Pareto improvement vs one-shot baselines.

Compositionality (A=COMPARE‑2D, B=PARITY‑4D):
- A→B and B→A MUST both yield 0 failures on both domains.
- Incremental KL from second repair MUST be ≤ 0.01 above max(first patches).

Non-enumerable COMPARE‑6D‑STRAT:
- Must be 0 failures on boundary strata.
- Must be ≥99.9% pass rate on certified interior set.
- Certificate MUST explicitly mark coverage-bounded (no global claim).

---

## 2. Core Method (what it is)
### 2.1 Spec satisfaction as constraints
Define correct answer token t*(x) from Spec(x).
Define margin:
m_φ(x) = logit_φ(t*(x)|x) - logit_φ(t¬*(x)|x)

Constraint threshold: τ = 1.0
Violation for a set D:
g(φ;D) = max_{x∈D} ReLU(τ - m_φ(x))

Feasible means g(φ;D)=0.

### 2.2 Collateral objective (minimality target)
Collateral loss:
L_col(φ) = E_{x∈RefBool-S} KL(p_θ(·|x) || p_{θ,φ}(·|x))

Regularizers:
R(φ) = λ2 Σℓ (||Uℓ||_F^2 + ||Vℓ||_F^2) + λ_grp Σℓ sqrt(||Uℓ||_F^2 + ||Vℓ||_F^2)

### 2.3 Constrained minimality optimization (Augmented Lagrangian)
We solve:
min_φ L_col(φ) + R(φ)  s.t. g(φ;D_spec)=0

Augmented Lagrangian:
L_AL(φ;λ,μ) = L_col(φ)+R(φ) + λ g(φ) + (μ/2) g(φ)^2

Schedule:
- init φ=0, λ=0, μ=1
- inner: Adam 2000 steps on L_AL
- update λ ← λ + μ g
- if g>0: μ ← 10μ else μ ← μ/2 (floor 1e-3)

Training uses a smooth approximation to max to avoid gradient brittleness:
Let v_i = ReLU(τ - m_φ(x_i)).
Use:
g_smooth = logsumexp(β v)/β with β=50 (fixed) over current minibatch, and run periodic full-set evaluation with true max.
Evaluation uses true max and true failure count.

### 2.4 CEGIS-style outer loop (non-optional)
- Maintain active constraint set D_spec (starts small).
- Solve constrained minimality on D_spec (ALM).
- Find counterexamples (violations) on full domain (exact) or coverage set (bounded).
- Add K hardest counterexamples (lowest margins), repeat until closure.

Key: This is not for sample efficiency; it is for achieving a better minimality point by focusing constraints where needed and keeping feasible set tight.

---

## 3. What makes it ICLR-grade (attack surface hardening)
### Attack A: "Just steering / LoReFT / adapter + hard mining"
We MUST demonstrate novelty at protocol level:
- Constrained minimality objective (not just improving a score).
- Output certificate artifacts with replay verification.
- Compositionality and interference analysis.

Killer evidence:
- Figure: feasible-only Pareto among methods at 0 failures (KL vs norm/layers).
- Table: ablations showing removing minimality or CEGIS worsens collateral at same feasibility.
- Compositionality matrix showing order effects and repair algebra (steering cannot claim).

### Attack B: "Certificates aren’t certificates"
We MUST not overclaim. We call them "replayable empirical certificates":
- Certificate includes exact scope, coverage, hashes, seeds.
- Verifier replays everything; any mismatch => FAIL.
- For coverage-bounded domains we explicitly report coverage and do not claim beyond it.

Killer evidence:
- Tamper tests: small perturbations or hash mismatches fail verification.
- Coverage report figure with per-stratum sample counts and error rates.

### Attack C: "Domains enumerable anyway; why CEGIS?"
We MUST show:
- Full-domain one-shot training can reach 0 failures but has worse collateral and/or larger patch.
- CEGIS + minimality reaches a better point on the feasible set.
- Additionally, COMPARE‑6D‑STRAT is non-enumerable so coverage matters.

Killer evidence:
- Trace plot comparing CEGIS vs OneShot in failures, KL, and patch norm over time.
- Non-enumerable coverage figure.

---

## 4. Determinism / Fail-Closed Rules (must implement)
- Every generator enumeration MUST be deterministic (fixed ordering).
- Every sampling uses fixed seeds recorded in certificate.
- Every metric uses deterministic decoding (greedy).
- Any missing/changed hash invalidates the certificate.
- If non-enumerable, certificate MUST declare coverage-bounded and include coverage plan hash; verifier MUST fail if coverage plan not reproduced.

---

## 5. What Codex must NOT do
- Must NOT invent new specs, gates, tokens, or metrics beyond SSOT without updating SSOT decision log.
- Must NOT change prompt wrapper formatting; it will break gate and comparability.
- Must NOT cherry-pick results; must run all defined specs including negative outcomes.
- Must NOT claim “proof”; must use “replayable empirical certificate” language.

---

## 6. Outputs required for a “10/10 paper”
At the end, the artifacts folder MUST contain:
- `certificate.json` per run
- `counterexamples.jsonl` per run
- `metrics.json` per run
- `plots/` containing the 5 signature figures
- `tables/` containing Table 1 and Table 2 CSVs
- `paper/` containing compiled PDF and LaTeX sources
- `MANIFEST.json` with file hashes

See `11_Progress_Tracker.md` for the session-by-session checklist.

