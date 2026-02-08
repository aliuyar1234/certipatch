# Progress Tracking, DONE Markers, and Session Logs

## Purpose
This document defines how Codex tracks progress session-by-session with zero drift and without searching the codebase.

---

# 1. STATUS.yaml is authoritative
All tasks are listed in `STATUS.yaml`.
Codex MUST update only STATUS.yaml to indicate progress.

State machine:
- TODO (aka NOT_STARTED) → IN_PROGRESS → DONE
- TODO (aka NOT_STARTED) → BLOCKED
- IN_PROGRESS → BLOCKED
- BLOCKED → IN_PROGRESS
- DONE is terminal (do not revert; create a new task ID if rework is needed).

Each task record MUST include:
- owner: "codex"
- start_date
- end_date (when DONE)
- artifacts: list of produced files/dirs
- evidence: command output or key metric
- commit_or_hash: git commit hash OR zip hash

---

# 2. Session logs
After each working session, create:
`session_logs/YYYY-MM-DD_sessionN.md`

Template MUST include:
- goals for session
- tasks attempted (IDs)
- what changed
- commands executed
- results (metrics)
- next steps

---

# 3. Paper artifact checklist (artifacts/DONE.md)
Mark each of:
- Figure 1..5
- Table 1..2
- Certificates for main runs
- Verifier PASS logs
- Compiled PDF

---

# 4. Minimal “Definition of Done” by subsystem
## Data
- Hash-stable generators.
- Disjointness checks pass.

## Patch
- Unit tests pass for identity, gate-off, gate-on.

## Optimizer
- Toy domain closes to 0 failures.

## CertiPatch
- Outer loop produces non-increasing failure counts; stops at closure.

## Certificates
- Verifier passes exact replay and fails tamper tests.

## Experiments
- Full run artifacts generated for each spec.

---

# 5. Troubleshooting rubric (if stuck)
- If failures won’t go to zero on COMPARE‑2D:
  1) verify answer tokenization
  2) verify you are reading logits at the correct position
  3) verify patch applies only at last position and only when gate=1
  4) increase μ via schedule (do not hack)
  5) check learning rate grid

- If KL is always ~0:
  - you are likely measuring collateral on gate=0 prompts. Fix RefBool-S wrapper.

- If verifier fails:
  - inspect hash mismatches first; do not relax verifier unless SSOT explicitly allows tolerance.

---

# 6. How to tell what runs are next
This file explains the *process*; the run queue itself lives in `STATUS.yaml`:
- Look under `phases.P7_experiment_runs` and execute tasks that are `TODO`/`IN_PROGRESS`.
- Primary driver (paper suite, resumable): `python scripts/reproduce_paper.py --config configs/paper_full.yaml --tier full --resume`.
- “Complete run” signal: `runs/<run_id>/{run_record.json,certificate.json,metrics.json,counterexamples.jsonl,patch.pt,report.html,MANIFEST.json}`.
- Live progress: tail `runs/<run_id>/{train_progress.jsonl,cegis_progress.jsonl,cex_progress.jsonl}` and watch `nvidia-smi -l 2`.
