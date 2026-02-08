"""certipatch.progress

Small utilities for progress/heartbeat logging during long runs.

Design goals:
- Zero impact on determinism (logging only).
- No schema changes: driven by cfg['_certipatch_runtime'].
- Safe defaults: enabled only when cfg.run.run_id is present.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    rt = cfg.get("_certipatch_runtime", {})
    return rt if isinstance(rt, Mapping) else {}


def progress_enabled(cfg: Mapping[str, Any]) -> bool:
    rt = _runtime(cfg)
    progress = rt.get("progress", {})
    if isinstance(progress, Mapping) and "enabled" in progress:
        return bool(progress.get("enabled"))
    run = cfg.get("run", {})
    return isinstance(run, Mapping) and bool(str(run.get("run_id", "")).strip())


def progress_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    rt = _runtime(cfg)
    progress = rt.get("progress", {})
    return progress if isinstance(progress, Mapping) else {}


def run_dir(cfg: Mapping[str, Any]) -> Optional[Path]:
    run = cfg.get("run", {})
    if not isinstance(run, Mapping):
        return None
    run_id = str(run.get("run_id", "")).strip()
    if not run_id:
        return None
    out_cfg = cfg.get("output", {})
    out_dir = str(out_cfg.get("out_dir", "runs")) if isinstance(out_cfg, Mapping) else "runs"
    return (Path(out_dir).resolve() / run_id).resolve()


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
