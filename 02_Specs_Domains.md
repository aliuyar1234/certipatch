# Specs and Domains (Exact Enumeration + Coverage-Bounded Plan)

## Common wrapper (must match exactly)
```
Instruction: Answer with a single token: Yes or No.
Question: {QUESTION_TEXT}
Answer:
```

- `{QUESTION_TEXT}` must have no leading/trailing whitespace.
- The full prompt must end with `Answer:` (no answer token appended during evaluation; model predicts it).
- The answer token is the next token after the colon.

## Token constraints
You MUST choose a token pair that is **single-token each** under the tokenizer.

### Primary attempt
- `t_yes = " Yes"`
- `t_no  = " No"`

### Deterministic fallback
If either is not a single token:
- `t_yes = " true"`
- `t_no  = " false"`

You MUST record which pair was used in the certificate, including token IDs.

### How to test tokenization (exact)
In HF tokenizers:
- Encode each string with `add_special_tokens=False`.
- Confirm length == 1.
- Additionally confirm that decoding the token ID yields exactly the original string (some tokenizers may normalize).

---

# Spec A — COMPARE‑2D (Enumerable; 10,000)
## Domain
Parameters:
- `a` in integers 0..99 represented as zero-padded 2-digit string.
- `b` in integers 0..99 represented as zero-padded 2-digit string.

Total domain size: 100 * 100 = 10,000.

## Canonical enumeration order (MUST)
```
for a in range(0, 100):
  for b in range(0, 100):
    yield (a,b)
```
Representation:
- `a_str = f"{a:02d}"`
- `b_str = f"{b:02d}"`

Prompt ID string: `compare2d/a={a_str}/b={b_str}`

## Question template (MUST)
`QUESTION_TEXT = "Is {a_str} greater than {b_str}?"`

Full prompt example:
```
Instruction: Answer with a single token: Yes or No.
Question: Is 07 greater than 42?
Answer:
```

## Labeler (MUST)
`label = (a > b)`.

For ties (a == b): label is No.

---

# Spec B — PARITY‑4D (Enumerable; 10,000)
## Domain
Parameter:
- `n` in integers 0..9999 represented as canonical decimal string: `n_str = str(n)` (NO zero padding).

Total domain size: 10,000.

## Canonical enumeration (MUST)
`for n in range(0, 10000): yield n`

Prompt ID: `parity4d/n={n}`

## Question template
`QUESTION_TEXT = "Is {n} even?"`

## Labeler
`label = (n % 2 == 0)`.

---

# Spec C — BALANCE‑PAREN‑14 (Enumerable; 32,767)
## Domain
All strings `s` over alphabet `{(,)}` with length `L <= 14`.

Domain size:
sum_{L=0..14} 2^L = 2^15 - 1 = 32,767.

## Canonical enumeration (MUST)
Order by increasing length, then lexicographic with '(' < ')'.
A deterministic way:

For each length L:
- Enumerate bitstrings b from 0..(2^L - 1)
- Map bit=0 -> '(' and bit=1 -> ')'
- This produces lexicographic order if you interpret bits in big-endian order.

Pseudo:
```
for L in range(0, 15):
  for b in range(0, 2**L):
    s = ''.join('(' if bit==0 else ')' for bit in bits_of_b_length_L_big_endian)
    yield s
```

Prompt ID:
- `balance14/L={L}/b={b}` OR include the string itself if safely escaped.

## Question template (MUST)
`QUESTION_TEXT = 'Is the parentheses string "{s}" balanced?'`

Example:
```
Instruction: Answer with a single token: Yes or No.
Question: Is the parentheses string "(()())" balanced?
Answer:
```

## Labeler (MUST)
Standard stack-based:
- counter = 0
- for ch in s:
  - if ch == '(' => counter++
  - else => counter-- ; if counter < 0 => unbalanced
- at end: balanced iff counter == 0 and never negative.

---

# Spec D — COMPARE‑6D‑STRAT (Non-enumerable; coverage-bounded)
## True domain
All pairs (a,b) where a,b are 6-digit integers 000000..999999 inclusive.
True size: 1e6 * 1e6 = 1e12 (not enumerable).

## Certified scope definition
The certificate scope is the BoolQA wrapper AND the **coverage set** defined by a fixed plan.

### Core concept: Most Significant Differing Digit (MSDD)
Represent a and b as 6 digits: d0 d1 d2 d3 d4 d5 (d0 most significant).
Define k = smallest index where a[k] != b[k]. If none, a==b.

Strata:
- S_k for k in {0,1,2,3,4,5}: pairs with equal prefix length k and differ at digit k.
- S_eq: a == b
- S_near: |a-b| in {1,2,5,10}
- S_ext: extreme values around 0 and 999999

### Certified coverage counts (MUST)
- For each S_k: N = 10,000
- S_eq: 5,000
- S_near: 10,000
- S_ext: 5,000
Total N_total = 60,000 + 5,000 + 10,000 + 5,000 = 80,000.

### Deterministic sampling plan (MUST)
You MUST define a deterministic generator for each stratum, parameterized by a fixed seed.

#### S_k generator
Goal: sample pairs that differ first at digit k.

Procedure for each sample:
1) Sample a shared prefix of length k: digits p0..p{k-1}.
   - Deterministic enumeration (NO RNG): let `P = 10^k`. For sample i, set `prefix_i = i % P` and take its base-10 representation zero-padded to length k to get digits p0..p{k-1}.
2) Choose two distinct digits for position k: da != db.
   - Deterministic mapping (NO RNG): define `pairs = [(da,db) for da in 0..9 for db in 0..9 if da!=db]` in lexicographic order (length 90). Use `(da,db) = pairs[i % 90]` for sample i.
3) Sample remaining suffix digits (k+1..5) independently for a and b.
   - Use fixed-seed PRNG; produce digits uniformly 0..9.

Then assemble:
- a = prefix + da + suffix_a
- b = prefix + db + suffix_b

This guarantees MSDD = k.

Pseudocode (S_k):
```python
pairs = [(da, db) for da in range(10) for db in range(10) if da != db]  # len=90, lex order
P = 10 ** k
rng = np.random.default_rng(seed + 1000*k)  # PCG64
for i in range(N):
    prefix_i = i % P
    prefix_digits = digits(prefix_i, k)            # list[int], zero-padded length k
    da, db = pairs[i % 90]
    suf_len = 6 - (k + 1)
    suf_a = rng.integers(0, 10, size=(suf_len,)).tolist()
    suf_b = rng.integers(0, 10, size=(suf_len,)).tolist()
    a_digits = prefix_digits + [da] + suf_a
    b_digits = prefix_digits + [db] + suf_b
    a = int(''.join(map(str, a_digits)))
    b = int(''.join(map(str, b_digits)))
    yield a, b
```


Implementation detail:
- For determinism and stable hashing, record the PRNG algorithm (e.g., Python `random.Random(seed)` or numpy PCG64). Use numpy `default_rng` (PCG64). Record numpy version in `run_record.json`/certificate and never change PRNG algorithm.

#### S_eq generator
Enumerate 5,000 values a in a deterministic spaced set:
- Use a = floor(i * 1e6 / 5000) for i=0..4999 (unique).
Set b=a.

#### S_near generator
For each delta in [1,2,5,10], generate 2,500 examples:
- Choose base b in deterministic spaced set (avoid overflow):
  - b = floor(i * (1e6 - 11) / 2500) for i=0..2499
- Set a = b + delta
Then also include negative deltas (a = b - delta) by swapping a,b deterministically, or define a separate block; MUST be consistent.

#### S_ext generator
Include 5,000 examples combining extremes:
- a in {000000,000001,000002,999997,999998,999999}
- b sampled from deterministic spaced set; and vice versa.

### Question template
`QUESTION_TEXT = "Is {a_str} greater than {b_str}?"` where a_str and b_str are 6-digit zero-padded.

### Labeler
`label = (a > b)`.

---

## Counterexample search beyond certified coverage (bounded; optional but recommended)
After passing the 80,000 certified coverage prompts, run a bounded search:
- 20,000 interior random samples (fixed seed).
- For the 200 lowest-margin points in the coverage set, perturb by ±1,±2,±5,±10 (clipped to [0,999999]) and evaluate.

These extra checks are recorded as "search" in the certificate but do NOT expand the certified scope unless you explicitly include them in coverage plan and hash.

---

## Non-overlap rule for reference suites
RefBool-S and RefBool-L prompts MUST be disjoint from all spec prompts by exact string match.

Recommended strategy:
- Use different question topics (e.g., geography, counting letters, simple string properties) that are not used in specs.
- Include the wrapper so gate=1 but questions differ.

Disjointness MUST be verified and logged in the manifest.
