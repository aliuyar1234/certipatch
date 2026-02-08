# JSON schemas (v1)

These schemas are normative references for implementing and validating:

- `certificate.json` (replayable empirical certificate)
- `run_record.json` (fully materialized configuration + environment + fingerprints)
- `metrics.json` (evaluation metrics)
- `config_schema.json` (YAML config surface after merge)

All schemas are JSON-Schema Draft-07 compatible.

Fail-closed verifier rule:
- The verifier MUST validate JSON artifacts against these schemas before trusting them.
- Any schema violation MUST cause verification failure.

Note:
- Repository integrity is enforced by `MANIFEST.sha256` (text file), not by a JSON schema.
