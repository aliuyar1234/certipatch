# Patch Family (GLR‑HP) and Hookpoint Integration

## Goal
Implement GLR‑HP: a gated low‑rank patch applied at a specific internal activation tensor for a frozen LM.

This doc answers the implementation questions Codex will otherwise ask:
- Which tensor exactly?
- At which position?
- How to gate?
- How to compose patches?
- How to compute parameter counts and norms?
- How to make this work both in TransformerLens and HuggingFace?

---

# 1. Activation choice: `resid_post` (post-block residual) at answer position
We define:
- Layer index ℓ in [0, n_layers-1]
- Position p = index of the **last token of the input prompt** (the token right after the colon in "Answer:" is predicted next).
- Residual stream vector at that layer and position.

### In TransformerLens (preferred)
TransformerLens hookpoints (for GPT‑2 style):
- `blocks.{l}.hook_resid_post`: residual stream after the whole block (attn+mlp+residual additions).

This is slightly broader than “post-MLP only”, but it is stable and standard. We standardize on `hook_resid_post` for simplicity and reproducibility.

**Patch location**:
- Hook function receives activation tensor of shape `[batch, seq, d_model]`.
- Patch applied to `act[:, p, :]` where `p = seq_len - 1`.

### In HuggingFace (fallback)
HuggingFace GPT‑2 model modules:
- Each block returns hidden states after attention + MLP.
- Hook the output of the block (`hidden_states`) and modify the last position.

Implementation options:
- Register forward hook on `model.transformer.h[l]` (the block module), capturing output hidden states.
- Carefully handle that HF returns a tuple; modify first element.

---

# 2. Gate function s(x)
Gate MUST be deterministic and **shared** across all specs and collateral suites.

Definition:
- Let `x_str` be the exact prompt string before tokenization.
- Normalize: `x_norm = x_str.rstrip()` (strip trailing whitespace only)
- s(x)=1 iff:
  1) `"Instruction: Answer with a single token: Yes or No."` appears as a substring in x_norm
  2) x_norm endswith `"Answer:"`

This gate is intentionally not spec-specific.

Implementation recommendation:
- Gate computed at batch construction time (per prompt) and stored as float tensor `[batch,1,1]` or `[batch,1]`.
- Avoid recomputing regex in the forward pass.

---

# 3. GLR‑HP operator: exact formula and stable implementation
For each selected layer ℓ:
- Parameters: Uℓ[d_model, r], Vℓ[d_model, r]
- For each example i and position p:
  - h = act[i, p, :]  # [d_model]
  - z = Vℓ^T h        # [r]
  - delta = Uℓ z      # [d_model]
  - act[i, p, :] += gate[i] * delta

Vectorized implementation for a batch:
- act_last: [batch, d_model]
- z = act_last @ Vℓ   # because V is [d_model,r]
- delta = z @ Uℓ^T    # since Uℓ is [d_model,r], Uℓ^T is [r,d_model]
- act_last += gate[:,None] * delta

Be explicit with shapes to avoid silent broadcasting bugs.

---

# 4. Patch composition (required)
You MUST support:
- A+B: apply two independent patches simultaneously.
- A→B: start from φ_A and learn Δφ_{B|A}; apply φ_A + Δφ.

Implementation requirement:
- A patch object MUST be able to export its parameters and be addable.
- Composition is parameter-wise addition for matching layers.

Recommended design:
- Represent patch parameters as dict:
  - `params[layer]["U"]`, `params[layer]["V"]`
- Define `Patch.__add__` returning new patch with U and V summed.

IMPORTANT: Composition is valid only if both patches share:
- same rank r
- same candidate layer list
- same hookpoint and gate definition

If not, treat as error.

---

# 5. Parameter count, norms, “effective layers”
### Parameter count
`param_count = Σℓ (d_model*r + d_model*r) = Σℓ 2*d_model*r`.

For GPT‑2 small:
- d_model=768, r=4, layers=4 => 24,576.

### Frobenius norm
Define:
`||φ||_F = sqrt( Σℓ (||Uℓ||_F^2 + ||Vℓ||_F^2) )`

### Effective layers
To support the minimality claims, compute per-layer magnitude:
`mag(ℓ) = sqrt(||Uℓ||_F^2 + ||Vℓ||_F^2)`

A layer is “effective” if `mag(ℓ) > 1e-3` (fixed threshold).
Report:
- effective_layers: list of layer indices
- num_effective_layers: len(list)

---

# 6. Group lasso penalty (exact)
We use:
`R_grp(φ) = λ_grp Σℓ mag(ℓ)`

This encourages a small number of layers to remain nonzero.

Implementation:
- Compute `mag(ℓ)` each step from parameters.
- Add to loss.

---

# 7. Hook correctness tests (mandatory)
You MUST implement these tests:

T1. Identity test (φ=0):
- For a fixed prompt, logits (full vocab) must match base exactly.

T2. Gate-off test:
- For a prompt that does NOT match gate, logits must match base exactly even with random φ.

T3. Gate-on test:
- For a gate-on prompt, logits must differ with random φ.

T4. Position test:
- Changing tokens after "Answer:" should not occur because prompt ends at "Answer:".
- Ensure p is the last index of input tokens.

---

# 8. Notes on where “Answer:” token is
The model predicts next token after the input sequence.
We treat the answer position p(x) as the index of the **last non-padding input token** in the prompt (the final token of the last line "Answer:"). In code, compute per example: `p = attention_mask.sum(dim=1) - 1` and index logits/activations at `p`.
Do NOT attempt to locate “Answer:” token index by string search after tokenization unless necessary.

Rule (robust to padding):
- Compute per-example `p = attention_mask.sum(dim=1) - 1` and gather logits/activations at those indices.

If (and only if) you enforce left padding everywhere, `p = seq_len-1` is equivalent, but do not rely on that assumption.

---

# 9. Common pitfalls and how to avoid them
1) Tokenization mismatch for Yes/No:
- Must check single-token. If not, fallback tokens.
- Report in certificate.

2) Hookpoint mismatch between TL and HF:
- TL uses standard names; HF modules differ.
- Use TL if possible; else create HF wrapper.

3) Accidentally patching the wrong position (padding bug):
- Prompts have variable length. In a padded batch, you MUST compute per-example answer indices `p = attention_mask.sum(dim=1)-1`.
- Apply the patch ONLY at those indices (not at a fixed `seq_len-1` if you use right padding).

4) Non-determinism from dropout:
- Set model to eval mode, disable dropout.

5) Mixed precision:
- If using BF16, be consistent; log dtype; ensure determinism.
