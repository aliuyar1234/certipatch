# FAQ / No-Questions Implementation Guide (Codex MUST read)

## Purpose
This is the “Codex will not ask follow-ups” document.
It lists every likely implementation question and answers it with exact decisions.

If Codex feels tempted to ask a question, it MUST look here first.

---

# Q1. Which library should I build on: TransformerLens or HuggingFace?
**Decision:** Prefer **TransformerLens** for hookpoint stability and simple patching. Provide HF fallback only if TL integration is blocked.

**Why:** TL exposes clean activation hookpoints with consistent naming and fewer edge cases than HF forward hooks.

**Action:**
- Implement a backend interface:
  - `Backend.forward_logits(prompts, patch=None)` returning logits at answer position.
  - `Backend.generate(prompts, patch=None, max_new_tokens=128)` for RefBool-L.

Then implement two backends:
- `TLBackend`
- `HFBackend`

---

# Q2. What exactly is the “answer position”?
**Decision:** The answer position **p(x)** is the index of the **last non-padding input token** in the prompt (the final token of the `"Answer:"` line). It MUST be computed **per example** from `attention_mask`:

- `p = attention_mask.long().sum(dim=1) - 1`  (shape `[B]`)

Then:
- answer logits are `logits[range(B), p, :]`.

**Rationale:** Prompts have variable length. If you batch with padding and assume `p = seq_len-1`, you will silently evaluate/pach on padding tokens for shorter prompts. Computing `p` from `attention_mask` is deterministic and backend-agnostic.

**Tokenizer requirement:** For GPT‑2, you MUST set:
- `tokenizer.pad_token = tokenizer.eos_token`
so padding produces a valid token ID and attention masks are correct.

**Do NOT** try to locate `"Answer:"` by searching token IDs or substrings. We only rely on the wrapper construction + `attention_mask`.

---

# Q3. How do I ensure the prompt ends with `"Answer:"` exactly?
**Decision:** The prompt builder MUST append a newline after Question line and then `Answer:` on the last line with no trailing content.

Canonical builder:
```
prompt = (
  "Instruction: Answer with a single token: Yes or No.\n"
  f"Question: {question_text}\n"
  "Answer:"
)
```
No trailing newline is required. During gate check, `rstrip()` may remove whitespace, but the emitted prompt must stay canonical.

---

# Q4. How do I implement the gate s(x) efficiently?
**Decision:**
- Gate computed from prompt strings at dataset construction time.
- Store `gate` as float tensor in the batch to avoid repeated string operations.

Implementation:
- `gate = 1.0` if wrapper substring exists AND prompt endswith "Answer:" after rstrip.
- Else 0.0

Unit tests MUST include:
- a correct wrapper prompt (gate=1)
- a prompt with different instruction text (gate=0)
- a prompt that contains wrapper but does not end with Answer: (gate=0)
- a prompt that ends with Answer: but missing wrapper (gate=0)

---

# Q5. How do I choose Yes/No tokens?
**Decision:** Choose the first pair that is single-token each:
1) `" Yes"` and `" No"`
2) else `" true"` and `" false"`

Test:
- encode string with add_special_tokens=False
- require len == 1
- require decode(token_id) == original string

Record:
- token strings
- token IDs
- tokenization mode (primary/fallback)

---

# Q6. How do I compute spec loss if we mainly use constraints?
**Decision:** Training objective uses ALM with g_smooth derived from margins, not cross-entropy. You still need to compute margins from logits.

Margins:
- If y=Yes: margin = logit_yes - logit_no
- If y=No:  margin = logit_no - logit_yes

Then v = relu(tau - margin).

No additional spec cross-entropy is required for CertiPatch (keep it minimal and crisp).
Baselines may use CE loss, but must be consistent and recorded.

---

# Q7. What exactly is g_true?
**Decision:**
- For a set D (active constraints), compute:
  - margin for every prompt
  - v = relu(tau - margin)
  - g_true = max(v)
Feasible iff g_true == 0.

For reporting:
- failures_count = number of prompts with predicted class != label
- violations_count = number of prompts with margin < tau

If you want strict feasibility in terms of margin, use violations_count == 0.

In the SSOT, “0 failures” refers to failures_count==0 on enumerable domains; report both.

---

# Q8. How do I implement g_smooth and why?
**Decision:** Use smooth approximation to max to get usable gradients.

Given batch v = relu(tau - margin):
- `g_smooth = logsumexp(beta * v) / beta` with beta=50.

Note:
- logsumexp is stable.
- Larger beta approximates max; 50 is a good fixed compromise.

During training:
- Use g_smooth for gradients.
During evaluation:
- compute g_true on full D_spec.

---

# Q9. How do I implement the augmented Lagrangian schedule without messing up?
**Decision:** Follow SSOT exactly; do not improvise.

State variables:
- λ (lambda)
- μ (mu)

Initialize: λ=0, μ=1.

Inner training:
- optimize φ for K=2000 steps on L_AL using current λ, μ.

After inner training:
- compute g_true on all D_spec
- update λ ← λ + μ*g_true
- if g_true > 0: μ ← 10*μ else μ ← max(1e-3, μ/2)

Stop conditions:
- you MAY early stop if g_true==0 and collateral plateau for 200 steps, but be deterministic about how you compute plateau (same evaluation interval and threshold).

---

# Q10. What is “collateral plateau” exactly?
**Decision:** Define plateau using a fixed evaluation window.

Example:
- Every 50 steps, compute running mean KL on a fixed subset of RefBool-S (e.g., first 1024 prompts).
- Keep best-so-far L_col.
- Plateau if best-so-far has not improved by ≥1e-5 for 200 consecutive steps.

This is deterministic if subset and intervals are fixed.

---

# Q11. How do I compute KL efficiently?
**Decision:** Correctness > micro-optimizations. Compute full-vocab KL at answer position in float32.

Vectorized steps:
1) logits_base: [B, V]
2) logits_patch: [B, V]
3) logp = log_softmax(logits_base, dim=-1)
4) logq = log_softmax(logits_patch, dim=-1)
5) p = exp(logp)
6) kl = sum(p*(logp - logq), dim=-1)
7) mean_kl = mean(kl)

Potential memory:
- B up to a few hundred is fine with V ~ 50k.
- Use smaller batch if GPU memory is tight; log batch size.

---

# Q12. Why KL(base||patched) not KL(patched||base)?
**Decision:** Use KL(base||patched) because base defines the reference behavior we want to preserve. Also it is stable and standard for “don’t deviate from base”.

---

# Q13. What about ΔPPL?
**Decision:** ΔPPL is optional; we will report ΔNLL on base argmax for simplicity and determinism. Long-form drift (RefBool-L) is the stronger collateral metric.

---

# Q14. How do I implement RefBool‑L long‑form drift (generation) correctly?
**Decision:** RefBool‑L MUST use **greedy decoding** with **cached past_key_values**, and the patch MUST be applied **only during the initial prompt forward pass** (where the answer position lives). Subsequent generation steps MUST NOT apply the patch (the patch’s effect propagates via the cached K/V states of the prompt).

## Generation algorithm (deterministic)
Given a batch of prompts:
1) Tokenize with padding; compute per-example answer positions `p = sum(attention_mask)-1`.
2) **Prompt pass (use_cache=True)**:
   - Base generation: run the base model (no patch) on the full prompt.
   - Patched generation: run the model on the full prompt **with patch enabled**, applying it at positions `p` across candidate layers.
   - Save `past_key_values` from this pass.
   - Compute the first generated token from logits at positions `p`.

3) **Autoregressive steps (t = 2..max_new_tokens)**:
   - Feed only the last generated token with `past_key_values` (use_cache=True).
   - **Patch MUST be disabled** in these steps.
   - Greedy choose next token from `logits[:, -1, :]`.
   - Stop per sequence when EOS is generated, else stop at `max_new_tokens`.

**Why this matters:** If you apply the patch to every generation step, you are no longer measuring drift from a *prompt-position hookpoint repair*; you are measuring a different intervention.

## Drift metrics (compare generated continuations only)
Let `gen_base` and `gen_patch` be the generated token ID sequences (excluding the prompt tokens), each truncated at EOS if produced, else length=max_new_tokens.

For each prompt:
- `divergence = 1` iff `gen_base != gen_patch` (exact token sequence mismatch), else 0.
- `first_diff_index`:
  - if identical: `max_new_tokens`
  - else: smallest index `i` where tokens differ (0-based). If one ends early, the first index past the shorter length counts as diff.
- `edit_distance`: token-level Levenshtein distance between `gen_base` and `gen_patch`.
- `normalized_edit_distance = edit_distance / max(len(gen_base), len(gen_patch))` (define 0 if both lengths 0).

---

# Q15. How do I ensure disjointness between spec prompts and collateral prompts?
**Decision:** Exact string match disjointness.

Implementation:
- Build a Python set of all spec prompt strings (for enumerable specs, this is manageable).
- When generating collateral prompts, assert `prompt not in spec_set`.
- If collision, skip and generate next prompt deterministically.

This must be logged.

---

# Q16. For COMPARE-6D-STRAT, what exactly is the certified set?
**Decision:** The certified set is the union of strata sets with fixed counts:
- S_k (k=0..5): 10,000 each
- S_eq: 5,000
- S_near: 10,000
- S_ext: 5,000
Total: 80,000

This set MUST be generated deterministically from the plan in `02_Specs_Domains.md`.
The certificate scope is coverage-bounded, and the coverage plan hash must be included.

---

# Q17. How should I store and iterate over large domains?
**Decision:**
- For enumerable 10k/32k domains, you can generate on the fly and evaluate in batches.
- You do NOT need to store all prompts to disk, but you MUST be able to reproduce them deterministically and compute domain_hash.

Approach:
- Implement generator that yields prompt strings in canonical order.
- For domain_hash, stream through generator and update sha256 incrementally (avoid holding all strings).

---

# Q18. How do I compute domain_hash without materializing all prompts?
**Decision:** Streaming hash.

Pseudo:
```
h = sha256()
for (id, prompt, label) in enumerate_domain():
    h.update(prompt.encode('utf-8'))
    h.update(b'\t')
    h.update(b'1' if label else b'0')
    h.update(b'\n')
domain_hash = h.hexdigest()
```

This must match verifier.

---

# Q19. How do I choose initial sample D_spec^(0) deterministically?
**Decision:** Use a deterministic permutation of indices.

Method:
- Determine domain size N (for enumerable).
- Create list indices 0..N-1.
- Shuffle with fixed seed using Fisher-Yates (Python random.Random(seed)).
- Take first n0 indices.
- D_spec^(0) is those prompts.

For balance14:
- N=32767, n0=2048.

For coverage-bounded set:
- treat the certified set as the “domain” and sample similarly if needed, but recommended to start with boundary strata to avoid trivial failure.

---

# Q20. What if the model is already perfect on COMPARE-2D (0 failures base)?
**Decision:** If base already has 0 failures, the spec is not useful.
Mitigation: we still keep COMPARE-2D as defined, but then claims shift:
- Use stricter τ margin constraints (e.g., τ=2.0) to create a nontrivial repair objective.
However SSOT fixed τ=1.0; do not change without decision log.

In practice, small LMs are usually imperfect; but if not, we will:
- increase difficulty by adding formatting variants (NOT allowed unless SSOT updated).
Therefore: if this happens, log it, update SSOT, and define a new compare spec with known base failures.

---

# Q21. How do I guarantee that CEGIS is “non-optional” in results?
**Decision:** Implement and run OneShot full-domain baselines that also achieve 0 failures, then show they have worse collateral / bigger patch norm.

Therefore:
- You MUST tune OneShot baselines to actually reach 0 failures if feasible under budget.
- If OneShot cannot reach 0 failures, the argument weakens. Increase budget uniformly across methods (fairly) or fix optimizer bug.

---

# Q22. How do I implement OneShot-FullDomain-MO fairly?
**Decision:**
- Train on the full domain as a dataset (minibatches).
- Loss: L_spec + α L_col + R.
- α grid fixed: [0, 0.01, 0.05, 0.1, 0.2]
- Choose smallest α that reaches 0 failures; then report its collateral.

If none reaches 0 failures, report best failure count and note as limitation.

---

# Q23. How do I implement OneShot-FullDomain-ALM baseline?
**Decision:**
- Use the ALM solver but with D_spec fixed to the full domain dataset (minibatches).
- No counterexample set growth.
- This isolates “CEGIS vs no CEGIS”.

---

# Q24. What is the difference between NoCEGIS ablation and OneShot-FullDomain-ALM?
**Decision:**
- NoCEGIS ablation: trains on a small initial sample only (n0), never adds counterexamples. Evaluated on full domain: demonstrates overfitting.
- OneShot-FullDomain-ALM: trains on full domain constraints but without the dynamic active-set focus: demonstrates minimality difference.

Both are needed.

---

# Q25. How to implement compositionality (A→B, B→A, A+B, Joint AB)?
**Decision:** You must produce six repaired systems:

1) A-only: run CertiPatch with Spec A only => φ_A
2) B-only: run CertiPatch with Spec B only => φ_B
3) A+B: apply φ_A + φ_B simultaneously at inference (no further training)
4) A→B: freeze φ_A; learn Δφ under constraints for both A and B; output φ_A + Δφ
5) B→A: symmetric
6) Joint AB: run a single CertiPatch run with combined constraints and objective.

Metrics:
- exact failures on A domain and B domain
- collateral KL and long-form drift
- patch norms

Expected:
- A+B likely breaks at least one spec (interference).
- A→B and B→A should restore 0/0 with bounded incremental KL.
- Joint may be best or comparable; order effects informative.

---

# Q26. How do I “freeze φ_A” in A→B?
**Decision:** Represent φ_total = φ_A + Δφ.
During training, keep φ_A parameters fixed (requires them as constants) and optimize only Δφ parameters.
In the forward pass, apply both (sum deltas).

---

# Q27. How do I ensure parameter budget remains matched in sequential repairs?
**Decision:** Δφ has the same parameterization as φ (same U,V shapes).
Yes, that doubles parameter count if you treat it as separate, but the sequential repair is allowed to add parameters because it represents a second patch. Still, for fairness, report parameter counts clearly:
- A→B has params for φ_A plus params for Δφ.
Also run Joint AB (single patch) for a fair comparison.

---

# Q28. How to avoid “cheating” with spec-specific gates?
**Decision:** Gate is wrapper-only; do not add spec-specific gating.
Spec-specific gating is prohibited because it trivializes compositionality and reduces collateral.

---

# Q29. How do I implement the verifier without missing anything?
**Decision:** Follow `06_Certificate_Spec.md`. Verifier must:
- recompute hashes
- rerun evaluation
- compare metrics
- fail closed

Figure 5 requires tamper tests.

---

# Q30. What is the run directory structure?
**Decision:** Every run creates a directory:
`runs/{run_id}/`
containing:
- certificate.json
- run_record.json
- patch weights file
- counterexamples.jsonl
- metrics.json
- logs/*.jsonl
- MANIFEST.json

No run is “successful” unless verifier passes.

---

# Q31. How do I generate figures/tables reliably?
**Decision:** Build a single script that reads metrics.json for all runs and outputs:
- figures as PNG/PDF in `paper/latex/figures/`
- tables as CSV in `paper/latex/tables/`
Then LaTeX includes them.

Figure semantics MUST match `05_Evaluation_and_Figures.md`.

---

# Q32. How do I pick the “best run” when sweeping hyperparameters?
**Decision:** Deterministic selection:
1) feasible (0 failures) is required for main claims
2) among feasible, choose minimal RefBool-S KL
3) if tie, choose smaller patch norm
Record chosen hyperparams in certificate and run_record.

---

# Q33. What if BALANCE-PAREN-14 fails to close under low collateral?
**Decision:** This is an allowed negative result and part of the paper’s honesty.
You MUST:
- output a fail-closed certificate with remaining failures and collateral curve
- include it in Table 1
- discuss as limitation

Do not drop the spec.

---

# Q34. What if LoRA baseline “wins” (lower KL at 0 failures)?
**Decision:** Then CertiPatch’s main claim about minimality may fail.
Mitigation:
- Ensure LoRA is trained under the same objective and evaluation.
- If LoRA genuinely wins, you still have novelty in certificates + compositionality + coverage-bounded scope.
But the paper positioning must shift: “CertiPatch as a protocol and artifacts; patch family not the key.”

Still, attempt to tune CertiPatch properly first; LoRA may be strong.

---

# Q35. What if g_true==0 but failures_count>0?
**Decision:** This can happen only if τ is too low or if margin calculation uses wrong tokens.
With τ=1.0, g_true==0 implies margins≥1.0, which implies correct class should win. If mismatch occurs:
- your margin computation is wrong (token IDs swapped) OR you computed logits at wrong position. Fix bug.

---

# Q36. What if failures_count==0 but g_true>0?
**Decision:** This means predictions are correct but margins < τ.
This is acceptable depending on how we define constraints.
In CertiPatch we treat constraint as margin≥τ. So you must push g_true to 0 as well for feasibility.
Report both metrics; ensure the constraint is the one used in ALM.

---

# Q37. How do I handle half precision?
**Decision:** Prefer float32 for determinism.
If using bf16 to speed up:
- Keep KL computation in float32.
- Record dtype in certificate.
- Increase verifier tolerance slightly (e.g., 1e-4).

Do not mix precision modes across runs you compare.

---

# Q38. How do I keep GPU compute under control?
**Decision:** Use batching and avoid repeated forward passes:
- For spec evaluation, run patched forward once; you do not need base forward.
- For collateral KL, you need both base and patched logits. You can compute base logits once and cache them for RefBool-S if memory allows; otherwise compute on the fly.

Option: cache base logits on disk for RefBool-S and reuse across runs; if you do, hash them and include in manifest.

---

# Q39. Where do I get wordlists for collateral suites without licensing issues?
**Decision:** Use the license-free wordlists shipped in `assets/wordlists/` (animals, cities, months, colors, substrings, general words).
Do not copy large copyrighted datasets.

---

# Q40. How do I ensure the project is “end-to-end paper ready”?
**Decision:** The definition is:
- All main runs completed with certificates and verifier PASS.
- Figures 1-5 generated and placed into LaTeX folder.
- Tables 1-2 generated and included.
- Paper skeleton compiled into PDF.

Use `artifacts/DONE.md` checklist and mark every item.

---

# Q41. Which reviewer attacks does each artifact neutralize?
**Decision:** Use this mapping:

A) “just steering”
- Figure 1 (minimality at feasibility)
- Figure 4 (compositionality/interference)
- Table 2 (ablation isolating minimality+CEGIS)

B) “certificates aren’t certificates”
- Certificate schema + verifier
- Figure 5 (tamper fail-closed)
- Figure 3 (coverage report)

C) “enumerable anyway; why CEGIS”
- Figure 2 (trace showing minimality benefit vs one-shot)
- Table 1 (one-shot baselines reach 0 failures but worse collateral)
- Figure 3 (non-enumerable coverage)

---

# Q42. What exact commands should Codex expect to run?
**Decision:** Provide a conceptual CLI. Implement something like:
- `python -m certipatch.run --config configs/compare2d_certipatch.yaml`
- `python -m certipatch.verify --run_dir runs/<run_id>`
- `python -m certipatch.make_figures --runs runs/ --out paper/latex/figures`
- `python -m certipatch.build_tables --runs runs/ --out paper/latex/tables`
- `latexmk -pdf paper/latex/main.tex`

Even if your actual CLI differs, keep these *semantic commands* available via scripts.

---

# Q43. What tests are absolutely required?
**Decision:** Minimum unit tests:
1) tokenization test for answer tokens
2) gate test
3) patch identity test (φ=0)
4) patch gate-off test
5) constraint margin correctness test
6) KL correctness test on toy logits
7) verifier PASS on toy run
8) verifier FAIL on tamper tests

---

# Q44. If I can only finish one thing first, what is the smallest end-to-end slice?
**Decision:** A toy run:
- COMPARE toy: a,b ∈ {0..19}
- RefBool-S small: 512 prompts
- One outer iteration
- Produce certificate and verify it

This validates everything: patching, metrics, ALM, hashing, certificates.

Then scale up.

---

# Q45. What if I need to change something not covered here?
**Decision:** You MUST:
- append to DECISION_LOG.md
- update 00_SSOT.md (new SSOT version)
- rerun affected experiments and regenerate artifacts

Do not make silent changes.
