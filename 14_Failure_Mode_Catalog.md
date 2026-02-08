# Failure Mode Catalog + Deterministic Debugging Steps

## Purpose
Catalog failure modes and precise debugging steps.
Codex should consult this before changing any design.

---

# Failure Mode 1: Base and patched logits differ even when φ=0
**Symptom:** identity test fails.
**Likely causes:**
- model not in eval mode (dropout)
- patch hook applied even when φ=0 due to in-place ops or nonzero init
- nondeterminism from CUDA kernels

**Fix steps:**
1) set model.eval()
2) ensure patch parameters initialized to exact zeros
3) ensure patch function returns activation unchanged when gate=0 or φ=0
4) enable deterministic algorithms and rerun

---

# Failure Mode 2: Gate-off prompts change under patch
**Symptom:** gate=0 but logits differ.
**Likely causes:**
- gate computed incorrectly (string mismatch)
- gate tensor broadcast wrong shape (e.g., gate becomes 1 due to dtype conversion)
- applying patch before gate multiplication

**Fix steps:**
- unit test gate on known prompts
- print gate values for a batch
- assert gate is 0.0 for gate-off prompts
- confirm multiplication occurs before addition

---

# Failure Mode 3: Spec evaluation uses wrong logits position
**Symptom:** training does not improve failures; margins nonsensical.
**Likely causes:**
- using logits at position p+1 or using last generated token rather than input position
- tokenizing prompt incorrectly (e.g., adding special tokens)

**Fix steps:**
- verify input_ids length and compute p = len-1
- ensure evaluation uses logits[:, p, :]
- confirm prompt ends with “Answer:” and no additional tokens

---

# Failure Mode 4: Token pair not single-token
**Symptom:** “ Yes” encodes into multiple tokens; predictions inconsistent.
**Fix steps:** use fallback tokens; record in certificate.

---

# Failure Mode 5: KL always near zero even for strong patches
**Symptom:** RefBool-S KL ~ 0 always.
**Likely causes:**
- collateral prompts are gate=0 (wrapper missing)
- computing KL on wrong position
- comparing patched vs patched, not base vs patched

**Fix steps:**
- assert gate=1 for RefBool-S prompts
- print a few prompts to confirm wrapper
- add sanity: KL(base||base)=0 and KL(base||random_patch)>0

---

# Failure Mode 6: ALM never reaches feasibility (g_true > 0 forever)
**Likely causes:**
- g_true computed incorrectly
- μ not increasing (schedule bug)
- learning rate too low
- patch applied to wrong tensor (no effect on logits)

**Fix steps:**
- log μ per inner round and ensure it grows when g_true>0
- test with higher lr in grid
- verify patch changes logits on a spec prompt when gate=1

---

# Failure Mode 7: CEGIS outer loop adds no counterexamples
**Likely causes:**
- counterexample search not running full sweep
- using wrong constraint definition (only misclass, not margin)
- forgetting to add selected examples to D_spec

**Fix steps:**
- print count of violations from sweep
- print top-5 hardest margins and ensure they are added
- assert D_spec size grows across iterations until closure

---

# Failure Mode 8: OneShot baselines cannot reach 0 failures but CertiPatch can
This is possible but weakens attack C defense. Investigate:
- Does OneShot get same number of steps as CertiPatch? It must.
- Is the OneShot objective correct?
- Is α sweep wide enough?

If still cannot, report but include analysis.

---

# Failure Mode 9: Compositionality A+B always works (0 failures)
Not a failure; it means interference is low. Still run A→B, B→A, Joint AB and report.

---

# Failure Mode 10: Verifier fails on exact replay
**Likely causes:**
- hashes computed with different newline normalization
- float tolerance too strict for dtype
- generator enumeration differs due to Python version differences (rare)

**Fix steps:**
- unify hashing functions and test on small examples
- set verifier tolerance according to dtype
- ensure canonical JSON serialization

---

# Failure Mode 11: Balance14 too easy or too hard
If base is already perfect, spec is not informative; if too hard, repair may be impossible under minimality.
In either case, report transparently and consider adding a second algorithmic spec with different structure ONLY with decision log update.

---

# Failure Mode 12: “Cheating” accusations about gate
Ensure you:
- measure collateral on gate=1 prompts (RefBool-S/L)
- include no-gate ablation showing collateral blow-up
- keep gate wrapper-only and shared across specs
