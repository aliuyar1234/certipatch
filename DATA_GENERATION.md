# DATA_GENERATION.md — Deterministic Offline Generators (Normative)

This document defines deterministic data generation procedures.

## 1) Shared BoolQA wrapper

All prompts MUST be constructed by:
- wrapper line from config
- `Question: {question}`
- suffix line from config

Spec generators MUST not alter the wrapper across domains.

## 2) Spec datasets

Generators are defined in `certipatch/specs/`.

### 2.1 compare_2d
- enumerate a in 0..99 and b in 0..99 in canonical order.

### 2.2 parity_4d
- enumerate n in 0..9999 in canonical order.

### 2.3 balance_paren_14
- enumerate length L = 0..14
- enumerate bitstrings i = 0..(2^L-1), mapping 0->'(' and 1->')'

### 2.4 compare_6d_strat_certified
Coverage plan parameters are read from config:
- per_msd_stratum_n, eq_n, near_n, ext_n, seed, near_deltas

The generator MUST:
- emit meta.stratum for each example
- verify each generated (a,b) satisfies its intended stratum
- fail on duplicates

## 3) Reference suites (collateral)

### 3.1 RefBool-S (distributional KL)
Generate deterministic prompts with BoolQA wrapper that are outside all spec templates.

Procedure (one valid choice; MUST be deterministic):
- Read wordlists:
  - animals, cities, months, colors, words
- Construct questions such as:
  - “Is the word '{w}' a color?” with label omitted (unlabeled suite)
  - “Does '{city}' start with the letter '{L}'?”
  - “Is '{month}' a month of the year?”
- Ensure these prompts do not match any spec question patterns; if they do, abort and adjust templates.

### 3.2 RefBool-L (long-form drift)
Generate prompts that request:
- Yes/No answer token
- One-sentence explanation

These prompts MUST still fire the gate, so the wrapper must be present.

Generation semantics are fixed by metrics definition:
- Patch applied only on prompt forward pass at answer position.
- Greedy decoding with cache, patch disabled on subsequent decode steps.

### 3.3 RefText (gate-off sanity)
Generate natural-text prompts that do not include the wrapper line, ensuring gate=0.

The verifier MUST confirm gate=0 for this suite; otherwise fail.

## 3) Canonical question templates (normative, hash-critical)

These templates MUST match exactly across:
- `certipatch/specs/*.py`
- domain hashing
- any certificate replay/verification

### 3.1 compare_2d
- Format: `a_str = f"{a:02d}"`, `b_str = f"{b:02d}"`
- `QUESTION_TEXT = "Is {a_str} greater than {b_str}?"`

### 3.2 parity_4d
- Format: `n_str = str(n)` (NO zero padding)
- `QUESTION_TEXT = "Is {n_str} even?"`

### 3.3 balance_paren_14
- `QUESTION_TEXT = "Is the parentheses string \"{s}\" balanced?"`
  - MUST use double quotes around `{s}`.
  - `{s}` consists only of '(' and ')' and has no whitespace.

### 3.4 compare_6d_strat
- Format: `a_str = f"{a:06d}"`, `b_str = f"{b:06d}"`
- `QUESTION_TEXT = "Is {a_str} greater than {b_str}?"`

FAIL-CLOSED:
- If any generator emits a different prompt string than these templates, the verifier MUST fail.
