# DATA_INVENTORY.md — Offline Data Assets and Hashing (Normative)

All data in this project is generated deterministically offline. No web access is required.

## 1) Generated datasets (outputs)

Runs MUST materialize datasets under `data/generated/<run_id>/`:

- `spec_compare_2d.jsonl`
- `spec_parity_4d.jsonl`
- `spec_balance_paren_14.jsonl`
- `spec_compare_6d_strat_certified.jsonl`
- `ref_refbool_s.jsonl`
- `ref_refbool_l.jsonl`
- `ref_reftext.jsonl`

Each JSONL line MUST contain:
- prompt (string)
- label (int) for spec datasets
- meta (object) with canonical parameters

Each dataset file MUST have a companion SHA256 file:
- `<name>.sha256` containing a single hex digest of the file content.

## 2) Wordlists (inputs)

Wordlists are stored under `assets/wordlists/` and are part of MANIFEST integrity.

They are used only for constructing deterministic reference prompts. They do not contain labels.

## 3) Hashing rules (canonical)

To ensure cross-machine stability:
- All text files MUST use UTF-8 encoding and LF line endings.
- JSON MUST be written with:
  - sorted keys
  - no trailing spaces
  - deterministic float formatting

The verifier MUST recompute dataset hashes and fail on mismatch.

