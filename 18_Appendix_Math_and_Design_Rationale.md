# Appendix: Math + Design Rationale (Constraint Minimality, CEGIS, Collateral)

## Purpose
Provide deeper mathematical and design rationale to strengthen the paper and to help implementation choices stay aligned with the intended theory.

This appendix is also a “don’t drift” guardrail: it explains why we chose each design element and what alternatives are explicitly rejected.

---

# A. Constraint choice: margin constraints vs classification correctness
We have two related notions of spec satisfaction:

1) **0 classification failures**:
\[
\hat y(x) = \mathrm{Spec}(x)\quad \forall x \in X
\]
This is a discrete condition based on argmax between two tokens.

2) **margin constraints**:
\[
m_\phi(x)=\mathrm{logit}_\phi(t^*(x)\mid x)-\mathrm{logit}_\phi(t_{\neg *}(x)\mid x)\ge \tau
\]
This is a stronger continuous condition.

## Why use margin constraints in optimization?
- Classification correctness is nondifferentiable and fragile; margin constraints provide a continuous surrogate that:
  - creates a “buffer” against tiny logit perturbations
  - improves robustness and makes feasibility stable across minor numerical differences
- Margin threshold τ=1.0 is a simple, fixed choice that is strong enough to matter but not too strong to be infeasible for small patches.

## Relation between margin and correctness
If τ>0, then satisfying the margin implies correctness, because:
- if m(x) >= τ > 0, the correct token has strictly larger logit than the incorrect one, hence argmax is correct.

Therefore, in ideal deterministic evaluation:
- violations_count==0 ⇒ failures_count==0.

If you observe violations_count==0 but failures_count>0, it indicates a bug (token mismatch or wrong position).

If failures_count==0 but violations_count>0, it indicates correct predictions with small margins; this is expected early in training.

---

# B. Why Augmented Lagrangian (ALM) instead of penalty-only or projection
We want:
\[
\min_\phi f(\phi) \ \text{s.t.}\ g(\phi)=0
\]
where:
- f(φ) = L_col(φ) + R(φ)
- g(φ) = max_{x∈D_spec} ReLU(τ - mφ(x))

## Penalty-only approach
A simple approach is:
\[
\min_\phi f(\phi) + \rho g(\phi)
\]
This often requires large ρ to enforce feasibility and can lead to poor conditioning.

## ALM approach
Augmented Lagrangian:
\[
\mathcal{L}_{AL}(\phi; \lambda,\mu)=f(\phi)+\lambda g(\phi)+\frac{\mu}{2}g(\phi)^2
\]
- λ acts like a dual variable, accumulating constraint pressure.
- μ increases only if feasibility is violated, enabling feasibility-first behavior.

ALM tends to reach feasibility more reliably than penalty-only, especially in nonconvex problems.

---

# C. Why smooth max (logsumexp) during training
Our constraint uses max over a set, which is nondifferentiable at ties and creates sparse gradients.

Given v_i = ReLU(τ - m(x_i)), the max is:
\[
g = \max_i v_i
\]

Smooth approximation:
\[
g_\beta = \frac{1}{\beta}\log\sum_i \exp(\beta v_i)
\]
Properties:
- g_\beta ≥ g
- g_\beta → g as β → ∞
- gradient distributes across multiple high-violation points, improving stability.

We fix β=50 to avoid another hyperparameter. This is sufficient to approximate max without extreme numerical overflow in float32 for typical v_i values.

---

# D. Why constraint generation (CEGIS-style) is about minimality, not sample efficiency
Even if X_spec is enumerable, training on all constraints uniformly can push the solution into regions that satisfy constraints but with unnecessary changes (higher collateral).

Constraint generation maintains an active set A⊂X_spec of violated or near-violated constraints.

Intuition:
- The feasible set is defined by the intersection of constraints. Most constraints are redundant once feasibility is achieved.
- Minimizing collateral subject to all constraints can be approximated by minimizing collateral subject to the active constraints (those that bind at optimum).
- CEGIS approximates this by repeatedly adding only constraints that are currently violated (or close to violated).

Therefore, CEGIS can converge to a solution closer to the constrained optimum with fewer “unnecessary” updates that would increase collateral.

This is the paper’s key methodological claim and is validated by OneShot full-domain baselines:
- if OneShot reaches 0 failures but has higher KL or higher norm, then CEGIS contributed to minimality.

---

# E. Why group lasso across layers
We want to show repairs can be localized (few effective layers), supporting interpretability and robustness.

Group lasso:
\[
R_{grp}=\lambda_{grp}\sum_\ell \sqrt{||U_\ell||_F^2 + ||V_\ell||_F^2}
\]
encourages entire layer patches to go to zero.

Benefits:
- fewer effective layers
- less interference across specs (hypothesis)
- clearer patch locality map in paper

Potential downside:
- might reduce expressivity; rank may need to be higher if group lasso too strong.

Hence we sweep λ_grp in a grid and report Pareto.

---

# F. Patch family rationale: why GLR‑HP is a good v1
We want a patch that is:
- small (few parameters)
- fast (negligible overhead)
- composable (additive)
- interpretable (low-rank, layer-sparse)
- implementable in hookpoints easily

GLR‑HP meets these requirements:
\[
h \leftarrow h + s(x)U(V^T h)
\]
It is essentially a low-rank linear operator applied at a single activation. It is more expressive than constant steering but far simpler than full adapters.

Crucially: we do not claim novelty from GLR‑HP alone; novelty is the protocol and certificate semantics.

---

# G. Collateral metrics: why both KL and long-form drift
KL at answer position is strong because:
- It measures distribution change at the certified decision point, where the spec is enforced.
- It is objective and continuous.

However, autoregressive generation can amplify small differences:
- a tiny KL at one step may lead to different token sampled/greedy chosen at subsequent positions.

Hence RefBool‑L:
- measures downstream divergence under greedy decoding in-scope.

Together, these make collateral claims harder to dismiss.

---

# H. Compositionality: why order effects matter
If patches are applied in shared scope, they can interfere.

Definitions:
- A+B: naive additive composition
- A→B: sequential repair that enforces both specs after applying A
- B→A: symmetric
- Joint AB: single repair enforcing both simultaneously

Order effects in A→B vs B→A indicate non-commutativity of repairs.
This is a novel “patch algebra” perspective and helps distinguish CertiPatch from simple steering or PEFT.

---

# I. Coverage-bounded certificates: why strata
For non-enumerable domains, random sampling misses boundary cases.
By stratifying based on MSDD and including boundary sets (eq, near, extremes), we:
- cover known hard regions systematically
- produce a coverage report that is meaningful and auditable
- can fail-closed with clear evidence of what was tested

This is essential to neutralize “certificates aren’t certificates” and “you only sampled easy points”.

---

# J. Minimality claim: what we can and cannot say
We can say:
- “CertiPatch approximately minimizes measured collateral on RefBool‑S among feasible solutions found under our patch family and budgets.”
- “CertiPatch achieves lower KL at 0 failures than full-domain one-shot baselines under matched budgets.”

We cannot say:
- global optimality
- formal minimality
- collateral bounds outside D_ref

This language must be enforced.

---

# K. Suggested appendix additions if space allows
- show additional Pareto curves for different λ_grp
- show per-layer magnitude maps
- include raw counterexample distributions (where base fails)

These increase paper credibility.