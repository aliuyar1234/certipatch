# Collateral Suites (RefBool‑S, RefBool‑L) + Hard-to-Dismiss Drift Metrics

## Purpose
Define collateral suites and metrics so that collateral claims are hard to dismiss.

Key requirement: collateral MUST be evaluated on prompts where the gate fires (same wrapper), otherwise KL is artificially near zero.

---

# 1. RefBool‑S (short distributional drift)
## 1.1 Suite definition
- Size: 20,000 prompts.
- All prompts must satisfy gate (wrapper).
- Prompts must be disjoint from all spec prompts by exact string match.

## 1.2 Generator (deterministic)
We use a pool of non-overlapping question families unrelated to numeric compare/parity/balance.

Question families (examples; implement 10 families to reduce overfitting):
1) String length parity:
   "Is the length of the string '{word}' even?"
2) Vowel count:
   "Does the word '{word}' contain more than 2 vowels?"
3) Alphabet order:
   "Is '{w1}' alphabetically before '{w2}'?"
4) Contains substring:
   "Does '{word}' contain the substring '{sub}'?"
5) Day/month mapping:
   "Is '{month}' in the first half of the year?"
6) Color membership:
   "Is '{color}' a primary color?"
7) Character case:
   "Is '{word}' all lowercase?"
8) Palindrome (short words):
   "Is '{word}' a palindrome?"
9) Set membership:
   "Is '{animal}' a mammal?"
10) Geography membership:
   "Is '{city}' in Europe?"

IMPORTANT:
- These are collateral prompts; we do NOT require ground truth accuracy, only stability.
- But prompts should be semantically plausible.

Deterministic construction:
- Use fixed wordlists stored in `assets/wordlists/` (small, license-free).
- Use canonical enumeration order across families, then fill until 20,000.
- Record generator code hash and suite hash.

## 1.3 Primary metric
Mean KL(base || patched) at answer position, full vocab.

Secondary:
- ΔNLL on base argmax token.

---

# 2. RefBool‑L (long-form drift)
This suite makes it harder to dismiss collateral with “KL is tiny but generation drift is large”.

## 2.1 Suite definition
- Size: 1,000 prompts.
- Gate must fire.
- Prompt asks for:
  - single-token Yes/No
  - one-sentence explanation
Example wrapper:
```
Instruction: Answer with a single token: Yes or No.
Question: Is a penguin a bird? Also give one sentence explaining your answer after the Yes/No.
Answer:
```

Implementation: keep wrapper constant and append clause to question.

## 2.2 Generation settings (fixed)
- Greedy decoding
- max_new_tokens=128
- temperature=0
- top_p not used
- stop at eos if produced, else at length.



**Patch application during generation (critical):** RefBool‑L MUST apply the patch only in the initial prompt forward pass (to modify the cached prompt activations at the answer position). Subsequent cached generation steps MUST run with the patch disabled. See FAQ Q14.
## 2.3 Metrics
- divergence_rate
- first_diff_index
- normalized token edit distance

---

# 3. RefText (gate=0 sanity)
- Size: 5,000 prompts.
- Default: synthetic natural-language sentences generated deterministically from the repo wordlists (license-free).
- Optional stronger variant: a small public text slice (e.g., Wikitext) if available; MUST record dataset fingerprint and license metadata.
- Ensure these prompts do NOT match the wrapper, so gate=0.
Expectation: patched outputs match base almost exactly. This clarifies scope.

---

# 4. Why this collateral setup is reviewer-hard
- KL at answer token captures distribution shift at the certified decision point.
- Long-form drift captures downstream autoregressive divergence.
- Gate-on evaluation ensures we are measuring in-scope drift, not trivial out-of-scope invariance.

---

# 5. Bootstrap CI (fixed)
For RefBool‑S:
- Use 2000 bootstrap resamples with fixed seeds [0..1999] or a fixed list.
- CI: 2.5% and 97.5% percentiles.

---

# 6. Disjointness checks (mandatory)
Before finalizing suites:
- Build a set of all spec prompts (strings).
- Assert each collateral prompt string is not in that set.
- Record counts and write a log line in run_record.json.
