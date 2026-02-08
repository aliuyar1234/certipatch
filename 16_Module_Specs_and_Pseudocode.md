# Module Specs + Function-Level Pseudocode (No Guesswork for Codex)

## Purpose
Define the software architecture and module responsibilities at the **spec level** so Codex can implement without guessing.

This is not a repo tree or zip packaging plan; it is a **behavioral spec** of modules, functions, and IO formats.

---

# 1. Core principle: pipeline is artifact-driven
Everything produces artifacts and hashes.
All downstream steps read artifacts, not internal state.

---

# 2. Conceptual modules (names are suggestions; behavior is mandatory)

## 2.1 config
Responsibilities:
- Load YAML overlay config and merge into configs/default.yaml (recursive dict merge; lists replaced).
- Validate config against schema.
- Freeze config into `run_record.json`.

Required functions:
- `load_config(path: str) -> dict`
- `resolve_inheritance(cfg: dict, base_dir: str) -> dict`
- `validate_cfg(cfg: dict) -> None`
- `freeze_cfg(cfg: dict) -> dict`  # ensures all defaults explicit

---

## 2.2 determinism
Responsibilities:
- Set seeds and deterministic flags.
- Provide `DeterminismReport` dict for certificate.

Required functions:
- `set_global_seed(seed: int) -> None`
- `enable_determinism() -> dict`  # returns flags used
- `get_env_snapshot() -> dict`    # python/torch/cuda/gpu

---

## 2.3 tokenization
Responsibilities:
- Determine answer tokens (primary vs fallback).
- Provide token IDs and checks.

Required functions:
- `choose_answer_tokens(tokenizer, primary_yes, primary_no, fallback_yes, fallback_no) -> dict`
Returns:
- `{"yes_str":..., "no_str":..., "yes_id":..., "no_id":..., "mode":"primary_yesno|fallback_truefalse"}`

---

## 2.4 prompts
Responsibilities:
- Build wrapper prompt.
- Gate predicate and gate tests.

Required:
- `build_prompt(question_text: str) -> str`
- `gate(prompt: str) -> bool`
- `gate_batch(prompts: list[str]) -> np.ndarray[float32]`  # 0/1

---

## 2.5 specs
Responsibilities:
- Enumerators and labelers for specs A/B/C and coverage plan for D.
- Canonical ordering and streaming hash.

Required:
- `iter_compare2d() -> iterator[(id, prompt, label)]`
- `iter_parity4d() -> iterator[...]`
- `iter_balance14() -> iterator[...]`
- `iter_compare6d_strat(seed: int) -> iterator[(stratum, id, prompt, label)]`
- `domain_hash(iterator) -> str` # sha256 streaming, prompt+label
- `coverage_plan_json(seed, counts, algorithm) -> dict` # used to hash plan

---

## 2.6 collateral_suites
Responsibilities:
- Generate RefBool-S, RefBool-L, RefText with hashes and disjointness.

Required:
- `build_refbool_s(n:int, spec_prompt_set:set[str]) -> list[str]`
- `build_refbool_l(n:int, spec_prompt_set:set[str]) -> list[str]`
- `build_reftext(n:int) -> list[str]`
- `suite_hash(prompts:list[str], extra:dict=None) -> str`

---

## 2.7 patch
Responsibilities:
- GLR-HP patch parameters, forward application, composition, metrics.

Required class:
`class GLRHPatch:`
- constructor takes: d_model, rank r, candidate layers list, hookpoint name, gate definition
- parameters: U[l], V[l]
- `apply(act: torch.Tensor, gate: torch.Tensor, pos: int) -> torch.Tensor` (vectorized)
- `param_count() -> int`
- `norm_fro() -> float`
- `layer_magnitudes() -> dict[layer->float]`
- `effective_layers(threshold=1e-3) -> list[int]`
- `save(path)`, `load(path)`
- `__add__(other)` for composition

---

## 2.8 backend
Responsibilities:
- Load model and run forward/generate with optional patch.

Required interface:
`class Backend:`
- `forward_logits(prompts: list[str], patch: GLRHPatch|None) -> torch.Tensor`  # returns logits at answer position [B,V]
  - MUST compute per-example answer indices `p = attention_mask.sum(dim=1) - 1` and gather logits as `logits[torch.arange(B), p]`.
- `forward_logits_with_hidden(prompts, patch)` if needed
- `generate(prompts, patch, max_new_tokens, greedy=True) -> list[list[int]]` tokens
  - MUST implement cached greedy decoding where the patch is applied only on the initial prompt pass (past_key_values=None) and disabled on subsequent generation steps.

Implementations:
- TLBackend (preferred)
- HFBackend (fallback)

---

## 2.9 metrics
Responsibilities:
- Compute failures, margins, KL, drift, complexity.
- No side effects.

Required:
- `spec_eval(logits, labels, yes_id, no_id, tau) -> dict`
- `kl_base_patched(logits_base, logits_patch) -> np.ndarray` per prompt
- `bootstrap_ci(values, seeds, alpha=0.05) -> (mean, lo, hi)`
- `drift_metrics(gen_base_tokens, gen_patch_tokens) -> dict`
- `complexity_metrics(patch) -> dict`

---

## 2.10 optimizer
Responsibilities:
- Solve constrained minimality on a fixed D_spec and D_ref.

Required:
- `solve_constrained_minimality(backend, patch_init, D_spec, D_ref, cfg, tokens) -> (patch, optim_log)`
Where:
- D_spec is list of (prompt, label)
- D_ref is list of prompts

Implementation MUST follow ALM schedule.

---

## 2.11 cegis
Responsibilities:
- Outer loop that grows D_spec by counterexamples.
- Counterexample search policies for enumerable vs coverage-bounded.

Required:
- `run_certipatch(spec_id, backend, cfg, tokens) -> run_dir`
- `find_counterexamples_enum(...)`
- `find_counterexamples_coverage(...)`
- `select_hardest(counterexamples, k_add)`

---

## 2.12 artifacts
Responsibilities:
- Write run_record, certificate, metrics, manifests.
- Compute file hashes.
- Provide deterministic serialization.

Required:
- `write_run_record(run_dir, cfg, env, tokens, ...)`
- `write_certificate(run_dir, certificate_dict)`
- `write_manifest(run_dir)`
- `load_run(run_dir)`

---

## 2.13 verifier
Responsibilities:
- Replay evaluation and confirm hashes and metrics.
- Fail-closed.

Required:
- `verify_run(run_dir) -> (pass:bool, report:dict)`
- `tamper_tests(run_dir, out_dir)`

---

## 2.14 paper_artifacts
Responsibilities:
- Read runs/, assemble figures and tables per spec.
- Output into paper/latex/figures and paper/latex/tables.

Required:
- `make_figures(runs_root, out_dir)`
- `make_tables(runs_root, out_dir)`

---

# 3. Canonical run IDs (mandatory)
Run IDs MUST be deterministic strings including:
- model name
- spec id
- method name (certipatch/baseline)
- seed
- timestamp optional but then include in run_record not in hashes; recommended: include timestamp in run_id but do not hash run_id.

Example:
`gpt2_compare2d_certipatch_seed0_2026-02-02T120000Z`

---

# 4. IO formats
## 4.1 metrics.json (required fields)
Store raw numbers used for plots:
- spec results: failures, margins
- collateral: KL array summary + CI
- drift metrics
- complexity metrics
- training trace (outer loop metrics per iter)

## 4.2 counterexamples.jsonl
One JSON object per outer iteration:
- outer_iter
- added_ids list
- added_margins list
- evaluation_summary (failures_count, min_margin, etc.)

---

# 5. CLI behavior (semantic)
You MUST provide scripts/entrypoints that map to:
- run one experiment from config
- verify a run
- build figures/tables
- build paper

Even if not called exactly these names, provide equivalent.

---

# 6. Deterministic testing harness
Implement `scripts/smoke_test.sh` conceptually:
- run toy compare
- verify certificate PASS
- run tamper tests => FAIL
- generate dummy figures/tables

This ensures end-to-end.
