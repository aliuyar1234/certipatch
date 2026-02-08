"""certipatch.determinism

Centralized determinism helpers used by both the reproduction driver and the verifier.

Goal: make CUDA/CPU runs as reproducible as practical while staying fail-closed when requested.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any, Dict, Mapping


def set_global_determinism(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply deterministic seeds/flags and return a JSON-serializable flag report.

    Notes:
    - `PYTHONHASHSEED` cannot be retroactively applied to an already-running interpreter; we record it.
    - CUDA determinism may additionally require setting `CUBLAS_WORKSPACE_CONFIG` before process start.
    """
    run = cfg.get("run", {}) if isinstance(cfg.get("run"), Mapping) else {}
    seeds = run.get("seeds", {}) if isinstance(run.get("seeds"), Mapping) else {}
    master = int(seeds.get("master", 0))
    np_seed = int(seeds.get("numpy", master))
    torch_seed = int(seeds.get("torch", master))
    deterministic_requested = bool(run.get("deterministic", True))

    if deterministic_requested and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        # Best-effort: cuBLAS determinism for some GEMMs. Prefer setting this before process start.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    flags: Dict[str, Any] = {
        "python_random_seed": master,
        "numpy_seed": np_seed,
        "torch_seed": torch_seed,
        "deterministic_requested": deterministic_requested,
        "pythonhashseed_env": os.environ.get("PYTHONHASHSEED", ""),
        "cublas_workspace_config_env": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }

    random.seed(master)

    try:
        import numpy as np

        np.random.seed(np_seed)
        flags["numpy_seed_applied"] = True
    except Exception as e:  # noqa: BLE001
        flags["numpy_seed_applied"] = False
        flags["numpy_error"] = str(e)

    try:
        import torch

        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

        flags.update(
            {
                "torch_version": getattr(torch, "__version__", "unknown"),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": str(getattr(getattr(torch, "version", None), "cuda", "") or ""),
                "cudnn_version": int(torch.backends.cudnn.version() or 0)
                if hasattr(torch.backends, "cudnn")
                else 0,
            }
        )

        if deterministic_requested:
            try:
                torch.use_deterministic_algorithms(True)
                flags["torch_deterministic_algorithms"] = True
            except Exception as e:  # noqa: BLE001
                flags["torch_deterministic_algorithms"] = False
                flags["torch_determinism_error"] = str(e)

            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                flags["cudnn_deterministic"] = True
                flags["cudnn_benchmark"] = False
            except Exception as e:  # noqa: BLE001
                flags["cudnn_flags_error"] = str(e)

            # Avoid TF32-induced numeric drift during replay verification.
            try:
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = False
                    flags["cuda_matmul_tf32"] = False
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = False
                    flags["cudnn_tf32"] = False
            except Exception as e:  # noqa: BLE001
                flags["tf32_disable_error"] = str(e)

            try:
                # PyTorch 2.x: "highest" disables TF32; "high"/"medium" may enable it.
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("highest")
                    flags["float32_matmul_precision"] = "highest"
            except Exception as e:  # noqa: BLE001
                flags["matmul_precision_error"] = str(e)
    except Exception as e:  # noqa: BLE001
        flags["torch_error"] = str(e)

    return flags


def collect_hardware_info() -> Dict[str, Any]:
    """Best-effort hardware snapshot for run_record.json (never raises)."""
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    try:
        import torch

        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
        if torch.cuda.is_available():
            gpus: list[Dict[str, Any]] = []
            for i in range(int(torch.cuda.device_count())):
                try:
                    props = torch.cuda.get_device_properties(i)
                    gpus.append(
                        {
                            "index": int(i),
                            "name": str(torch.cuda.get_device_name(i)),
                            "total_memory_bytes": int(getattr(props, "total_memory", 0)),
                            "capability": f"{int(getattr(props, 'major', 0))}.{int(getattr(props, 'minor', 0))}",
                            "multi_processor_count": int(
                                getattr(props, "multi_processor_count", 0)
                            ),
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    gpus.append({"index": int(i), "error": str(e)})
            info["gpus"] = gpus
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = str(e)

    return info
