"""certipatch.config

This project is driven by YAML configs under `configs/` and validated by
`schemas/config_schema.json`.

Design goals:
- Flexible: any local model is allowed if it satisfies the adapter contract.
- Deterministic: seeds + coverage plans + canonical enumeration must be fixed.
- Fail-closed: if required config keys are missing or inconsistent with schemas,
  the run MUST abort rather than guess.

This scaffold file defines the expected high-level config surface and a minimal
loading/validation contract. It is intentionally not a full implementation.

Codex MUST implement:
- `load_config(path)` that merges default config and an overlay config.
- `validate_config(cfg)` using JSON Schema, failing if validation fails.
- `freeze_config(cfg)` that writes a fully resolved run_record.json capturing
  all resolved defaults and computed derived values (e.g., candidate layers).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml
from jsonschema import Draft7Validator

ConfigDict = Dict[str, Any]

_META_KEY = "_certipatch_meta"


def _json_pointer(path: Sequence[Any]) -> str:
    if not path:
        return "/"

    def escape(token: str) -> str:
        return token.replace("~", "~0").replace("/", "~1")

    parts: list[str] = []
    for item in path:
        if isinstance(item, int):
            parts.append(str(item))
        else:
            parts.append(escape(str(item)))
    return "/" + "/".join(parts)


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge dicts; lists/scalars replace."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged: dict[str, Any] = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _read_yaml(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"YAML file not found: {path}") from e
    except Exception as e:  # noqa: BLE001 - fail-closed with context
        raise ValueError(f"Failed to parse YAML: {path}: {e}") from e

    return {} if data is None else data


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"JSON file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {path}: {e}") from e


def _sha256_file(path: str | os.PathLike[str]) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info(repo_root: str) -> Dict[str, Any]:
    git_dir = Path(repo_root) / ".git"
    if not git_dir.exists():
        return {"is_git_repo": False}

    def run_git(args: list[str]) -> str:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return "unknown"
        return proc.stdout.strip()

    return {
        "is_git_repo": True,
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_porcelain": run_git(["status", "--porcelain"]),
    }


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def load_config(config_path: str, default_path: str = "configs/default.yaml") -> ConfigDict:
    """Load an overlay YAML and merge it into the default config.

    Merge semantics (MUST):
    - Load default first, then overlay.
    - Dict keys merge recursively.
    - Lists are replaced (not concatenated).
    - Scalars are replaced.

    Fail-closed rules (MUST):
    - If any YAML file cannot be parsed, abort.
    - If the merged config violates the JSON schema, abort.

    Returns:
        A merged dict suitable for downstream modules.

    Pseudocode:
        default = yaml.load(default_path)
        overlay = yaml.load(config_path)
        cfg = deep_merge(default, overlay)
        validate_config(cfg)
        return cfg
    """
    default = _read_yaml(default_path)
    overlay = _read_yaml(config_path)

    if not isinstance(default, dict):
        raise ValueError(f"Default config must be a mapping, got: {type(default).__name__}")
    if not isinstance(overlay, dict):
        raise ValueError(f"Overlay config must be a mapping, got: {type(overlay).__name__}")

    cfg: ConfigDict = _deep_merge(default, overlay)
    validate_config(cfg)

    # Stash resolved paths and invocation info for run_record generation.
    cfg.setdefault(_META_KEY, {})
    if isinstance(cfg[_META_KEY], dict):
        cfg[_META_KEY].update(
            {
                "default_path": default_path,
                "overlay_path": config_path,
                "command": " ".join(sys.argv) if sys.argv else "",
            }
        )

    return cfg


def validate_config(
    cfg: Mapping[str, Any], schema_path: str = "schemas/config_schema.json"
) -> None:
    """Validate cfg against the config schema.

    MUST:
    - Validate using Draft-07 JSON schema.
    - Provide a human-friendly error message with the failing JSON pointer.
    - Abort on first failure.

    """
    schema = _read_json(schema_path)
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(cfg), key=lambda e: (list(e.absolute_path), e.message))
    if not errors:
        return

    err = errors[0]
    pointer = _json_pointer(list(err.absolute_path))
    raise ValueError(f"Config validation failed at {pointer}: {err.message}")


def freeze_run_record(cfg: Mapping[str, Any]) -> ConfigDict:
    """Return a run_record dict that is fully resolved and hashable.

    The run_record MUST include:
    - Fully resolved candidate layers (explicit list).
    - Resolved answer token mode (primary or fallback).
    - Model fingerprint information returned by the adapter.
    - Derived coverage plan hashes for non-enumerable specs.
    - The SHA256 of MANIFEST.sha256.

    The run_record MUST be written to `runs/<run_id>/run_record.json`.

    """
    if "run" not in cfg or not isinstance(cfg["run"], Mapping):
        raise ValueError("cfg['run'] must exist and be a mapping")
    run_id = str(cfg["run"].get("run_id", "")).strip()
    if not run_id:
        raise ValueError("cfg['run']['run_id'] is required")

    out_dir = "runs"
    if isinstance(cfg.get("output"), Mapping):
        out_dir = str(cfg["output"].get("out_dir", out_dir))

    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    repo_root = os.getcwd()
    meta = cfg.get(_META_KEY, {}) if isinstance(cfg.get(_META_KEY), Mapping) else {}
    config_path = str(meta.get("overlay_path", "unknown"))
    command = str(meta.get("command", " ".join(sys.argv)))

    seeds = cfg["run"].get("seeds", {})
    if not isinstance(seeds, Mapping):
        raise ValueError("cfg['run']['seeds'] must be a mapping")

    objective_cfg = cfg.get("objective", {}) if isinstance(cfg.get("objective"), Mapping) else {}
    g_smooth_formula = (
        str(objective_cfg.get("g_smooth_formula", "log_mean_exp")).strip() or "log_mean_exp"
    )

    manifest_path = Path(repo_root) / "MANIFEST.sha256"
    manifest_sha256 = _sha256_file(manifest_path) if manifest_path.exists() else "missing"

    config_clean: ConfigDict = dict(cfg)
    config_clean.pop(_META_KEY, None)

    try:
        from certipatch.determinism import collect_hardware_info
    except Exception:  # noqa: BLE001
        collect_hardware_info = None  # type: ignore[assignment]

    run_record: ConfigDict = {
        "schema_version": "1.0",
        "run_id": run_id,
        "config_path": config_path,
        "config": config_clean,
        "command": command,
        "git": _git_info(repo_root),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": repo_root,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": collect_hardware_info()
            if collect_hardware_info is not None
            else {"error": "unavailable"},
        },
        "seeds": dict(seeds),
        "objective_resolved": {"g_smooth_formula": g_smooth_formula},
        "manifest_sha256": manifest_sha256,
    }

    out_path = run_dir / "run_record.json"
    out_path.write_text(_canonical_json(run_record), encoding="utf-8")
    return run_record
