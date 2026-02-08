# Paper Writing Plan (ICLR Voice, Attack-Hardened, No Guesswork)

## Purpose
This folder contains a LaTeX *plan* (not full writing) that Codex can use to produce a strong ICLR-style paper with minimal thinking.

This document defines:
- exact section outline
- what to put in each section
- where figures/tables go
- what claims are allowed (and prohibited)
- “attack-surface hardening” language snippets

---

# 1. Paper structure (ICLR style)
## 1.1 Title
Use final title from SSOT:
**CertiPatch: Minimal-Collateral Specification Repair for Language Models via Counterexample-Guided Hookpoint Patches**

## 1.2 Abstract (already finalized in SSOT)
Copy from `00_SSOT.md` unless experiments force slight edits.

## 1.3 Section outline (must follow)
1. Introduction
2. Problem: Spec Repair with Collateral Minimality
3. Method: CertiPatch
4. Replayable Empirical Certificates
5. Experiments
6. Discussion, Limitations, Ethics
7. Related Work
8. Conclusion
Appendices: implementation details, certificate schema, additional plots

---

# 2. Introduction: required content
Must include:
- Motivation: editing vs repair, side effects, lack of auditable artifacts.
- Thesis: treat editing as *specification repair* with minimal collateral.
- One paragraph on “empirical certificates not proofs”.
- Bullet contributions.

**Must include the 3 reviewer hardening statements:**
A) Our contribution is a constraint-first protocol and certificate semantics, not a novel steering vector.
B) Certificates are replayable evaluation artifacts; no formal guarantee is claimed.
C) CEGIS is required to reach better minimality points at feasibility and to handle non-enumerable coverage-bounded scopes.

---

# 3. Problem formalization section (include math)
Define:
- Spec domain X_spec, labeler Spec(x), gate s(x)
- Answer tokens, prediction, margin
- Collateral suite D_ref, KL metric
- Patch family GLR-HP
- Objective: minimize collateral s.t. g(φ)=0

Explicitly state:
- For enumerable domains, “0 failures” is exact.
- For non-enumerable, we certify only coverage-bounded sets.

---

# 4. Method section (CertiPatch)
## 4.1 Patch family (GLR-HP)
Include formula:
h ← h + s(x) U(V^T h)

## 4.2 Constrained optimizer (Augmented Lagrangian)
Include ALM objective and schedule (μ, λ updates).
State: feasibility-first then minimality refinement.

## 4.3 Counterexample-guided constraint generation
Explain active set:
- initialize small subset
- solve constrained minimality
- search counterexamples, add hardest
- repeat to closure

Emphasize: not sample efficiency; **active constraints improve minimality at feasibility**.

---

# 5. Certificates section
Define:
- certificate.json schema (high-level)
- scope types
- fail-closed verification
- tamper tests

Include small example snippet (not too long).
State explicitly:
- Not a proof; only a replayable empirical certificate.

---

# 6. Experiments section (must match figure/table plan)
## 6.1 Experimental setup
- models and revisions
- prompt wrapper
- answer tokens
- patch parameter budget
- training protocol and seeds
- hyperparameter grid

## 6.2 Specs and domains
- COMPARE-2D
- PARITY-4D
- BALANCE-PAREN-14
- COMPARE-6D-STRAT coverage plan (strata)

## 6.3 Baselines
- OneShot-FullDomain-MO
- OneShot-FullDomain-ALM
- SoftPrompt
- LoRA rank-4 on attn.c_proj for the 4 candidate layers (budget-matched)
- Steering baselines

## 6.4 Results: minimality at feasibility
Place Figure 1 + Table 1.
Narrative MUST be:
- At 0 failures, CertiPatch yields lower KL and/or lower norm vs baselines.

## 6.5 Results: CEGIS necessity
Place Figure 2 + Table 2.
Narrative MUST be:
- One-shot training can reach feasibility but not the same minimality frontier.
- No-CEGIS ablation fails or worse.

## 6.6 Results: coverage-bounded certificates
Place Figure 3 and a short paragraph:
- show boundary strata results
- emphasize scope-bounded claims

## 6.7 Results: compositionality
Place Figure 4.
Narrative:
- Naive additive composition can interfere.
- Sequential repair restores both specs with bounded incremental collateral.
- Order effects quantify interference.

## 6.8 Results: certificate replay and tamper tests
Place Figure 5.

---

# 7. Discussion, limitations, ethics
Must include:
- Certificates are empirical and scope-bounded.
- Patches could be misused to enforce undesired policy changes; mitigation is transparency (certificate + patch publication).
- Failure cases: BALANCE may not close under strict collateral; must report.

---

# 8. Prohibited claims (do not write)
- “provably” or “guaranteed” outside certified scope
- “formal verification” unless you actually integrate formal methods (you are not)
- “bounded collateral” as a theorem; only as measured on D_ref

---

# 9. Required phrasing snippets (copyable)
## On certificates
“We output a replayable empirical certificate: a hash-tied record of scope, coverage, counterexamples, and measured drift that a verifier can deterministically replay. This is not a formal proof; out-of-scope behavior is explicitly uncertified.”

## On why not just steering
“While our patch parameterization is deliberately simple, our contribution is a constraint-first repair protocol that optimizes minimal collateral drift subject to satisfaction and emits auditable certificates; these properties are not provided by steering or PEFT baselines.”

## On CEGIS
“CEGIS is used as constraint generation: it concentrates optimization on violated constraints and enables better minimality at feasibility compared to one-shot full-domain training, even when enumeration is possible.”

---

# 10. Figure captions (templates)
Provide exact captions aligned with attack hardening.
(See `05_Evaluation_and_Figures.md` for plot semantics.)

---

# 11. LaTeX skeleton files (provided in paper/latex/)
- main.tex compile-ready minimal skeleton
- sections/*.tex for each section
- tables/*.tex or tables/*.csv integration
- references.bib file present (may be empty; no external lookups)

Codex SHALL fill text from this plan and insert figures generated by the pipeline.