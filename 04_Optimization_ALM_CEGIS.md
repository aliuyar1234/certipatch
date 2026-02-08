# Optimization: Augmented Lagrangian Minimality + CEGIS Loop

## Goal
Implement the constrained minimality solver and outer CEGIS loop exactly as the paper requires.

This document is intentionally explicit (math + engineering details) to minimize Codex “thinking time”.

---

# 1. Objective recap (constraint-first)
We solve:
minimize collateral + regularization
subject to zero spec violations on certified scope.

Let D_spec be the current active constraint set (CEGIS maintains it).
Let D_ref be RefBool-S.

## 1.1 Spec margin
For each prompt x, with correct answer token t*(x), define:
m_φ(x) = logit_φ(t*(x)|x) - logit_φ(t_neg(x)|x)

with t_neg being the other token in {t_yes, t_no}.

## 1.2 Feasibility threshold
Fixed τ = 1.0.

Violation scalar for a set:
g_true(φ; D_spec) = max_{x∈D_spec} ReLU(τ - m_φ(x))

Feasible iff g_true == 0.

---

# 2. Collateral loss (RefBool-S)
Compute mean KL at answer position:
KL(p||q) where:
- p = softmax(logits_base)
- q = softmax(logits_patched)

Define:
L_col(φ) = mean_x KL(pθ(.|x) || pθ,φ(.|x))

Implementation detail:
- Compute logits at answer position for full vocab.
- Use stable log-softmax.
- KL(p||q) = sum_i p_i * (log p_i - log q_i)

Secondary collateral metrics (for reporting):
- ΔNLL_base_argmax = -log q(t_base_argmax) - (-log p(t_base_argmax)).

---

# 3. Regularizers
R(φ) = λ2 Σℓ (||Uℓ||_F^2 + ||Vℓ||_F^2) + λ_grp Σℓ mag(ℓ)
where mag(ℓ)=sqrt(||Uℓ||_F^2+||Vℓ||_F^2)

---

# 4. Augmented Lagrangian solver (ALM)
We minimize:
L_AL = L_col + R + λ g + (μ/2) g^2

Where g is a differentiable proxy for constraint violation.

## 4.1 Smooth violation proxy for training
Because g_true uses max over set, gradients can be brittle.
Training uses:
v_i = ReLU(τ - m_φ(x_i))

Given a batch B from D_spec:
g_batch_smooth = logsumexp(β v_i)/β
with β=50 (fixed).

Optionally, to reduce sensitivity to batch size, you may use:
g_batch_smooth = (1/β)*log(mean(exp(β v_i))).
But you MUST keep it consistent once chosen (record in config).
Implementation: select via `objective.g_smooth_formula` ("log_mean_exp" | "logsumexp"); recorded in `runs/<run_id>/run_record.json`.

Evaluation uses g_true on full D_spec (or full domain).

## 4.2 ALM schedule (fixed)
Initialize:
φ = 0
λ = 0
μ = 1.0

For each outer iteration of the *inner solver*:
1) Run Adam for K=2000 steps minimizing L_AL (using g_batch_smooth for gradients).
2) Compute g_true over D_spec (full sweep) in eval mode.
3) Update:
   λ ← λ + μ * g_true
   if g_true > 0:
       μ ← 10 * μ
   else:
       μ ← max(1e-3, μ / 2)

Stopping inside an outer CertiPatch iteration:
- You MAY stop early if:
  - g_true == 0 AND
  - L_col has not improved by ≥1e-5 for 200 steps.
But ensure determinism: use fixed checkpointing intervals.

CEGIS integration:
- Warm-start the inner solver across CEGIS outer iterations by carrying `(λ, μ)` forward (deterministic).

---

# 5. CertiPatch outer loop (CEGIS)
CEGIS maintains an active set D_spec.

## 5.1 Initialization
For each spec:
- Choose n0:
  - For 10k domains: n0 = 512
  - For balance14 (32,767): n0 = 2048
- D_spec^(0) = deterministic sample from domain.
Sampling MUST be:
- fixed seed
- without replacement
- canonical ordering; take indices from a deterministic shuffle.

## 5.2 Counterexample search policies
Enumerable spec:
- Evaluate all X_spec.
- Collect counterexamples:
  Cex = {x: prediction != label} OR {x: margin < τ} (use margin constraint, not just misclassification).
Prefer to define counterexample as margin violation: v_i > 0.

Coverage-bounded spec:
- Evaluate certified coverage set; treat it as “X_cert”.
- Counterexamples defined on X_cert only.

## 5.3 Hardest-margin selection (default)
From Cex, select K_add examples with smallest margins m_φ(x).
Tie-break lexicographically by prompt ID.

Typical K_add:
- 256 for 10k domains
- 512 for balance14
- 512 for coverage-bounded if violations appear

Add them to D_spec; repeat.

## 5.4 Closure
Stop when:
- Enumerable: no margin violations on full X_spec (exact closure).
- Coverage-bounded: no margin violations on certified coverage set X_cert.
Certificate MUST reflect which.

---

# 6. Why CEGIS buys minimality (and how to demonstrate)
Key comparison:
- OneShot-FullDomain can reach feasibility but may require:
  - larger patch norm
  - more effective layers
  - higher KL collateral
because it trains uniformly across constraints and can “over-correct” easy regions.

CEGIS produces a smaller active set focusing on hard constraints, allowing:
- tighter feasibility enforcement with less collateral drift.

You MUST implement and report:
- At equal feasibility (0 failures), compare KL and patch norm between:
  - CertiPatch
  - OneShot-FullDomain-MO
  - OneShot-FullDomain-ALM (no CEGIS)

---

# 7. Engineering details (to avoid bugs)
## 7.1 Batch mixing
You will likely need to compute L_col and g in the same step.
Recommended:
- Each step uses one batch from D_ref and one batch from D_spec.
- Total loss = mean KL on ref batch + regularizer + λ g_spec + (μ/2) g_spec^2
- g_spec computed from spec batch.

## 7.2 Evaluation intervals
- Every N steps (e.g., 200), compute:
  - g_true on all of D_spec
  - running mean KL on a fixed subset of D_ref for speed
- Do full D_ref evaluation only at the end of each outer iteration.

## 7.3 Numerical stability
- Use float32 for KL computations even if model runs in bf16.
- Use `log_softmax` for logits.
- Clamp probabilities if needed (but prefer stable log computations).

## 7.4 Deterministic dataloading
- Do not use multi-worker dataloaders unless deterministic is guaranteed.
- Use fixed ordering (no shuffle) for evaluation.

---

# 8. Pseudocode (full)
```
function SolveConstrainedMinimality(phi_init, D_spec, D_ref, config):
    phi ← phi_init
    λ ← 0
    μ ← 1
    for inner_round in 1..config.max_inner_rounds:
        optimizer ← Adam(phi, lr=config.lr)
        for step in 1..K:
            batch_spec ← next_batch(D_spec)
            batch_ref  ← next_batch(D_ref)

            logits_base_ref   ← fθ(batch_ref)
            logits_patch_ref  ← fθ,phi(batch_ref)
            L_col ← mean_KL(logits_base_ref, logits_patch_ref, answer_pos)

            logits_patch_spec ← fθ,phi(batch_spec)
            margins ← compute_margins(logits_patch_spec, labels)
            v ← relu(τ - margins)
            g_smooth ← logsumexp(β v) / β

            R ← l2 + group_lasso(phi)
            L ← L_col + R + λ*g_smooth + (μ/2)*g_smooth*g_smooth

            backprop(L); optimizer.step()

        g_true ← max_{x in D_spec} relu(τ - margin(x))
        λ ← λ + μ * g_true
        if g_true > 0: μ ← 10μ else μ ← max(1e-3, μ/2)
        if g_true == 0 and collateral plateau: break
    return phi
```

---

# 9. Sanity tests you MUST run before real experiments
- Toy COMPARE with range 0..19 should reach 0 failures within ≤2 outer iterations.
- With gate disabled (s=0), training must not change anything (KL ~ 0).
- With α=0 (no collateral), patch should achieve feasibility but collateral likely rises; this supports claims and should match ablation expectations.
