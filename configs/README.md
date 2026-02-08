# Configs

Configs are **overlays**.

Rule:
1) Load `configs/default.yaml`.
2) Load the run-specific config (e.g. `compare2d_certipatch.yaml`).
3) Recursively merge dictionaries (run-specific values override defaults).

This keeps small configs readable and prevents duplication.

A run MUST record:
- the path to the overlay config,
- the fully-materialized merged config (in `run_record.json`) for replay.
