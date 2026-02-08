# Implementation recipes (to prevent Codex drift)

This file is intentionally concrete: it gives “how-to” pseudocode for the hardest parts.

---

## R1. Tokenization and answer positions (ALL methods)
MUST do:
1) `tokenizer = AutoTokenizer.from_pretrained(model_name, revision=...)`
2) If tokenizer has no pad token (e.g., GPT‑2): `tokenizer.pad_token = tokenizer.eos_token`
3) Tokenize:
```python
enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
input_ids = enc["input_ids"]            # [B,S]
attention_mask = enc["attention_mask"]  # [B,S], 1 for real tokens, 0 for pad
p = attention_mask.long().sum(dim=1) - 1  # [B] last non-pad token index
```
4) Gather answer logits:
```python
logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B,S,V]
answer_logits = logits[torch.arange(B), p]  # [B,V]
```

---

## R2. CertiPatch patch application at `resid_post` (HF backend)
Target: GPT‑2 style `transformer.h[l]` blocks.

### Hook strategy
Register a forward hook on each selected block `h[l]` that edits the block output hidden states **at positions p**.

HF GPT‑2 block output is a tuple:
`(hidden_states, present, attentions?, cross_attentions?)`.
You MUST replace `hidden_states` and return a new tuple.

### Patch math
For each selected layer ℓ:
- parameters: `U_ℓ ∈ R^{d×r}`, `V_ℓ ∈ R^{d×r}`.
- given `h ∈ R^{B×d}` at the answer positions:
  - `delta = (h @ V_ℓ) @ U_ℓ.T`  # shape [B,d]
- apply gated add:
  - `h ← h + gate[:,None] * delta`

### Hook pseudocode
```python
def make_block_hook(patch, layer_idx):
    def hook(module, inputs, output):
        if not patch.enabled:
            return output
        if patch.prompt_only and patch.past_is_not_none:
            return output  # disable during generation steps
        hidden = output[0]  # [B,S,d]
        B, S, d = hidden.shape
        p = patch.current_p  # [B] on device
        gate = patch.current_gate  # [B] float (0 or 1)
        h = hidden[torch.arange(B), p]             # [B,d]
        U, V = patch.U[layer_idx], patch.V[layer_idx]
        delta = (h @ V) @ U.T                      # [B,d]
        hidden = hidden.clone()
        hidden[torch.arange(B), p] = h + gate[:,None] * delta
        return (hidden,) + output[1:]
    return hook
```

**MUST:** The backend MUST set `patch.current_p` and `patch.current_gate` before every forward call.

---

## R3. RefBool‑L generation (cached; patch only in prompt pass)
Goal: generate continuations while applying the patch only during the initial prompt forward call.

### Greedy generation with caching (batched)
```python
# 1) prompt pass
patch.enabled = True
patch.prompt_only = True
patch.past_is_not_none = False
out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
answer_logits = out.logits[torch.arange(B), p]  # [B,V]
next_tok = answer_logits.argmax(dim=-1)         # [B]

past = out.past_key_values

# 2) generation steps (patch disabled)
patch.enabled = False
gen = []
for t in range(max_new_tokens):
    gen.append(next_tok)
    out = model(input_ids=next_tok[:,None], past_key_values=past, use_cache=True)
    next_tok = out.logits[:, -1, :].argmax(dim=-1)
    past = out.past_key_values

gen = torch.stack(gen, dim=1)  # [B,T]
```

Stop-on-EOS:
- track a `done` mask; once EOS produced, keep producing EOS (or keep last token) for determinism, and truncate in metric computation.

Compute drift metrics on `gen` vs `gen_base`.

---

## R4. SoftPrompt baseline (HF)
Train only a learned prefix embedding `P ∈ R^{k×d}`.

### Forward pass
```python
P = nn.Parameter(0.02 * torch.randn(k, d_model))
tok_emb = model.transformer.wte(input_ids)      # [B,S,d]
P_batch = P[None,:,:].expand(B, -1, -1)         # [B,k,d]
inputs_embeds = torch.cat([P_batch, tok_emb], dim=1)      # [B,S+k,d]

attn2 = torch.cat([torch.ones(B,k,device=...), attention_mask], dim=1)  # [B,S+k]
p2 = attn2.long().sum(dim=1) - 1

out = model(inputs_embeds=inputs_embeds, attention_mask=attn2, use_cache=False)
answer_logits = out.logits[torch.arange(B), p2]  # [B,V]
```

**MUST:** Freeze all model weights; only P has gradients.

---

## R5. LoRA baseline (HF + PEFT)
Recommended: use `peft` library.

Pseudocode:
```python
from peft import LoraConfig, get_peft_model

cfg = LoraConfig(
    r=4, lora_alpha=8, lora_dropout=0.0,
    target_modules=["attn.c_proj"],  # GPT-2
    bias="none", task_type="CAUSAL_LM"
)
lora_model = get_peft_model(base_model, cfg)
# Freeze base weights handled by PEFT wrapper; train only LoRA params.
```

Limit to candidate layers:
- easiest: create module name filters that include only those block indices, e.g. `transformer.h.3.attn.c_proj`, ...
- MUST be deterministic and logged.

---

## R6. CEGIS counterexample selection (deterministic)
Given a set of counterexamples `C` with margins:
1) sort by ascending margin (hardest first).
2) tie-break by example ID (lex order of (a,b) or string prompt).
3) take first K.

---

## R7. Hashing (sha256; stable)
Define helper:
```python
def sha256_bytes(b): return hashlib.sha256(b).hexdigest()

def sha256_text(s): return sha256_bytes(s.encode("utf-8"))
```

Domain hash rule:
- iterate prompts in deterministic order,
- for each: append `prompt + "\n" + label + "\n"`,
- hash the concatenated UTF‑8 bytes.

Suite hash rule:
- same but without label (or include label if suite has programmatic labeler; MUST be defined).

---

This file is an implementation companion to SSOT.
- If any doc contradicts `00_SSOT.md`, **SSOT wins**.
- For implementation details not explicitly decided in SSOT, this file is the default reference.
