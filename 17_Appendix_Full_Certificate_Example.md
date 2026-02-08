# Appendix: Full Certificate Example (Replayable Empirical Certificate)

## Example certificate.json (annotated)

Below is a representative example. In real runs, all hashes are real sha256 strings.

```json
{
  "schema_version": "1.0",
  "run_id": "EXAMPLE_gpt2_compare2d_certipatch_seed0_2026-02-02T120000Z",
  "timestamp_utc": "2026-02-02T12:00:00Z",
  "model": {
    "model_name": "gpt2",
    "model_revision": "PINNED_COMMIT_HASH_HERE",
    "tokenizer_name": "gpt2",
    "answer_tokens": {
      "yes_token": " Yes",
      "no_token": " No",
      "tokenization_mode": "primary_yesno",
      "yes_token_id": 3763,
      "no_token_id": 645
    }
  },
  "patch": {
    "patch_family": "GLR-HP",
    "rank_r": 4,
    "hookpoint": "resid_post",
    "candidate_layers": [
      3,
      6,
      9,
      11
    ],
    "gate_definition": {
      "gate_name": "boolqa_wrapper_v1",
      "gate_predicate_text": "contains wrapper line AND endswith Answer: after rstrip()",
      "gate_code_hash": "<sha256-of-gate-code>"
    },
    "patch_parameter_count": 24576,
    "patch_norm_fro": 0.123456,
    "effective_layers": [
      11
    ],
    "patch_weights_hash": "<sha256-of-patch-file>"
  },
  "specs": [
    {
      "spec_id": "compare2d",
      "spec_version": "1.0",
      "enumerable": true,
      "domain_size": 10000,
      "domain_generator": {
        "generator_name": "compare2d_v1",
        "generator_code_hash": "<sha256>",
        "domain_hash": "<sha256-of-domain-prompts-and-labels>"
      },
      "labeler_code_hash": "<sha256>",
      "certified_scope": {
        "scope_type": "exact_enumeration",
        "coverage_plan_hash": null
      },
      "evaluation": {
        "failures": 0,
        "pass_rate": 1.0,
        "min_margin": 1.234,
        "p05_margin": 2.345
      }
    }
  ],
  "cegis": {
    "outer_iterations": 4,
    "initial_sample_n0": 512,
    "counterexample_add_k": 256,
    "counterexample_policy": "hardest_margin",
    "counterexample_sets": {
      "cex_jsonl_hash": "<sha256-of-counterexamples-jsonl>",
      "cex_count_total": 912
    },
    "search": {
      "search_policies": [
        "exact_sweep"
      ],
      "search_budgets": {
        "max_evals_per_iter": 10000,
        "max_total_evals": 40000
      },
      "search_seeds": [
        0,
        1,
        2,
        3
      ]
    }
  },
  "objective": {
    "constraint_margin_tau": 1.0,
    "collateral_metric": "kl_base_to_patched_answer_token",
    "regularizers": {
      "lambda_l2": 0.0001,
      "lambda_group": 0.001
    },
    "augmented_lagrangian": {
      "mu_init": 1.0,
      "mu_mult_on_violation": 10.0,
      "mu_div_on_feasible": 2.0,
      "inner_steps_per_outer": 2000
    }
  },
  "collateral": {
    "ref_suites": [
      {
        "suite_id": "RefBool-S",
        "suite_hash": "<sha256>",
        "n_prompts": 20000,
        "mean_kl": 0.00123,
        "bootstrap_ci_95": [
          0.0011,
          0.00135
        ]
      },
      {
        "suite_id": "RefBool-L",
        "suite_hash": "<sha256>",
        "n_prompts": 1000,
        "divergence_rate": 0.12,
        "mean_first_diff_index": 47.8,
        "mean_norm_edit_distance": 0.08
      },
      {
        "suite_id": "RefText",
        "suite_hash": "<sha256>",
        "n_prompts": 5000,
        "mean_kl": 0.0
      }
    ]
  },
  "reproducibility": {
    "seeds": {
      "python": 0,
      "numpy": 0,
      "torch": 0
    },
    "determinism_flags": {
      "torch_deterministic": true,
      "cudnn_deterministic": true
    },
    "environment": {
      "python_version": "3.11.x",
      "torch_version": "2.x",
      "cuda_version": "12.x",
      "device": "NVIDIA ..."
    }
  },
  "fail_closed": {
    "claims": [
      "exact_zero_failures_only_if_enumerable",
      "coverage_bounded_no_full_domain_claims",
      "no_out_of_scope_guarantees"
    ],
    "verification_tolerances": {
      "max_abs_metric_diff": 1e-06
    }
  }
}
```

### Annotation highlights
- `domain_hash` binds the exact prompt formatting and labels.
- `coverage_plan_hash` is null here because scope is exact enumeration.
- `patch_weights_hash` detects tampering; Figure 5 modifies patch weights to force FAIL.
- `suite_hash` binds the collateral prompts and generation settings.
- `verification_tolerances` must be strict when float32 deterministic.

### What the verifier recomputes
1) model revision and tokenizer; 2) answer token IDs; 3) domain and suite hashes; 4) exact evaluation; 5) collateral metrics.
Any mismatch => FAIL.
