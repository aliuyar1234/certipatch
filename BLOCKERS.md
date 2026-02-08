# BLOCKERS.md

## Active blockers
- None.

## Resolved blockers
- **2026-02-04 — ALM solver state carry-over:** `SolverState.inner_round` was being reused as a start index in
  `certipatch/cegis/trainer.py`, so once it reached `optimizer.max_inner_rounds`, later CEGIS outer iterations
  skipped ALM entirely (`g_true=null`) and the patch stopped updating. Fixed by always running
  `range(0, max_inner_rounds)` per call and treating `inner_round` only as a deterministic round offset.
  Regression test: `tests/test_trainer_alm.py::test_solve_constrained_minimality_does_not_skip_when_state_inner_round_at_limit`.

## Notes (not blockers)
- Full-tier collateral eval is slow (especially RefBool-L generation). For debugging, reduce:
  - `data.refbool_l_n`, `evaluation.generation.max_new_tokens`, and/or `data.reftext_n`.
