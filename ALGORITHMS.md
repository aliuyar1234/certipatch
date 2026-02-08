# ALGORITHMS.md — ALM, CEGIS, Coverage, Compositionality (Normative)

This document defines the exact algorithms CertiPatch MUST implement.

## 1) Constrained minimality objective (repair semantics)

Given:
- frozen model `f_θ`
- patch parameters `φ`
- active constraint set `D_spec`
- reference set `D_ref` (gate-firing prompts)

Define:
- margin for labeled prompt `x`:
  `m_φ(x) = logit_φ(t_correct|x) - logit_φ(t_incorrect|x)`
- constraint threshold `τ` (config key: `objective.tau_margin`)
- max-violation function:
  `g(φ; D_spec) = max_{x ∈ D_spec} max(0, τ - m_φ(x))`

Collateral objective:
- `L_col(φ) = E_{x ∈ D_ref} KL(p_θ(.|x) || p_{θ,φ}(.|x))` at answer position.

Complexity penalty:
- L2 + group sparsity across layers (config keys under `regularizers`).

The repair problem is:
- minimize `L_col(φ) + R(φ)` subject to `g(φ; D_spec) = 0`.

Feasibility is binary:
- On enumerable specs, feasibility means zero failures on the full domain sweep.
- On coverage-bounded specs, feasibility means zero failures on certified coverage sets.

## 2) Augmented Lagrangian method (ALM) schedule

Use augmented Lagrangian:
- `L_AL(φ; λ, μ) = L_col(φ) + R(φ) + λ g(φ) + (μ/2) g(φ)^2`.

Schedule (config keys under `alm`):
- initialize: `λ=0`, `μ=mu_init`
- warm-start across CEGIS outer iterations: within a single run, carry `(λ, μ)` forward between repeated
  calls to the inner solver (recorded in run_record for auditability).
- inner optimize `φ` for `inner_steps_per_outer` Adam steps on `L_AL`
- evaluate full-set `g_full = g(φ; D_spec)`
- update:
  - `λ ← λ + μ g_full`
  - if `g_full > 0`: `μ ← max(mu_floor, μ * mu_mult_on_violation)`
  - else: `μ ← max(mu_floor, μ / mu_div_on_feasible)`

Stop inner rounds:
- stop if `g_full == 0` and collateral does not improve for a fixed patience window (documented in run_record).

Important:
- Feasibility MUST be checked on the full active set, not minibatches.
- The system MUST NOT silently relax τ. If a run cannot reach feasibility, it MUST fail-close and report.

## 3) CEGIS loop (counterexample-guided constraint generation)

Inputs:
- `X_spec` spec domain (enumerable or coverage-bounded)
- initial sample size `n0`
- per-iteration additions `k_add`
- counterexample policy: hardest-margin (default)

Algorithm:
1) Initialize `D_spec` from deterministic sampling of `X_spec` (or from the first `n0` canonical examples).
2) Repeat for at most `max_outer_iters`:
   a) Run ALM solver on current `D_spec` to produce `φ`.
   b) Find counterexamples under `φ`:
      - enumerable: exact sweep
      - coverage-bounded: certified coverage evaluation + bounded additional search budgets
   c) If no counterexamples are found within scope and budgets, stop.
   d) Add `k_add` hardest counterexamples to `D_spec` (lowest margin, lexicographic tie-break).
3) Emit artifacts: counterexample set history + active set hash + certificate.

CEGIS is non-optional by design:
- It is used to achieve constrained minimality with a compact active set.
- The paper requires a baseline that trains on the full enumerable domain once and reaches feasibility but with worse collateral or larger patch.

## 4) Coverage plan for non-enumerable domain (COMPARE-6D-STRAT)

Certified scope is a deterministic set consisting of strata:
- MSDD strata S_k for k in 0..5
- equality stratum S_eq
- near stratum S_near
- extremes stratum S_ext

Exact generation rules are specified in `certipatch/specs/compare_6d_strat.py`.

The certificate MUST include:
- per-stratum counts
- per-stratum failure rates
- a coverage plan hash derived from the plan parameters and generator code hash

The verifier MUST treat absence of counterexamples outside certified scope as:
- “no counterexamples found within certified coverage and bounded budgets”.

## 5) Compositionality / interference experiment

Specs:
- A: compare_2d
- B: parity_4d

Conditions:
1) A-only: learn φ_A on A
2) B-only: learn φ_B on B
3) A+B: apply φ_A and φ_B simultaneously (additive at hookpoints)
4) A→B: freeze φ_A; learn Δφ that satisfies A and B; output φ_A + Δφ
5) B→A: symmetric
6) Joint AB: single run with union constraints; output φ_AB

Report:
- failures on A and B domains (exact)
- RefBool-S KL, RefBool-L drift
- patch complexity (#effective layers, fro norm, parameter count)
- order sensitivity: compare A→B vs B→A.
