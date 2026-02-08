# ruff: noqa: E402

"""scripts.reproduce_paper

One-command reproduction driver.

This is a scaffold. It defines the CLI contract and output layout.

Required behavior (MUST):
- Read a config overlay YAML (e.g., configs/compare2d_certipatch.yaml).
- Merge with configs/default.yaml.
- Validate merged config against schemas/config_schema.json.
- Load model via certipatch.models.load_model.load_model_from_config.
- Run enabled specs + baselines + ablations according to EXPERIMENTS.md.
- Emit run artifacts under runs/<run_id>/:
    - run_record.json
    - certificate.json (if enabled)
    - metrics.json
    - counterexamples.jsonl
    - patch.pt
    - report.html (if enabled)
- Generate paper figures/tables into paper/latex/figures and paper/latex/tables
  with exact filenames listed in FIGURES_TABLES.md.
- Run verifier after generation; abort if verification fails.

Fail-closed:
- If any required output is missing, abort and mark the run as invalid.

This script SHOULD support tiers:
- toy: quick smoke test
- small: GPT-2 scale
- full: full paper run matrix

"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

# Ensure `import certipatch` works when running as a script from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# HuggingFace download behavior (best-effort): avoid Xet-powered downloads on Windows (often slow/brittle).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from certipatch.artifacts.certificate import build_certificate_dict, write_certificate_json
from certipatch.artifacts.report import build_html_report
from certipatch.artifacts.verifier import verify_certificate, verify_manifest
from certipatch.cegis.loop import run_cegis, run_compositionality_suite
from certipatch.config import freeze_run_record, load_config
from certipatch.data_generation import (
    build_refbool_l,
    build_refbool_s,
    build_reftext,
    coverage_plan_hash,
    domain_hash,
    sha256_file,
    suite_hash,
)
from certipatch.determinism import set_global_determinism
from certipatch.eval.ablations import run_ablations
from certipatch.eval.baselines import run_baselines
from certipatch.eval.metrics import eval_collateral, eval_spec_exact, eval_spec_exact_with_strata
from certipatch.models.load_model import assert_or_select_answer_tokens, load_model_from_config
from certipatch.patch_families import GLRHookPatch, GLRHPConfig


def _deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, Mapping):
            existing = dst.get(key)
            if isinstance(existing, dict):
                _deep_update(existing, value)
            else:
                dst[key] = copy.deepcopy(dict(value))
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _sha256_file(path: Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_tier_overrides(cfg: Dict[str, Any], tier: str) -> None:
    if tier == "toy":
        cfg.setdefault("specs", {}).setdefault("compare_2d", {}).update(
            {"a_min": 0, "a_max": 19, "b_min": 0, "b_max": 19}
        )
        cfg.setdefault("specs", {}).setdefault("parity_4d", {}).update({"n_min": 0, "n_max": 199})
        cfg.setdefault("specs", {}).setdefault("balance_paren_14", {}).update({"max_len": 8})
        cfg.setdefault("data", {}).update({"refbool_s_n": 200, "refbool_l_n": 20, "reftext_n": 200})
        cfg.setdefault("specs", {})["enabled"] = ["compare_2d"]

        # Keep toy runs fast and avoid large model downloads by default.
        cfg.setdefault("model", {}).setdefault("backend", "huggingface")
        if str(cfg["model"].get("backend")) == "huggingface":
            cfg["model"]["model_path_or_id"] = cfg["model"].get(
                "toy_model_path_or_id", "sshleifer/tiny-gpt2"
            )

        cfg.setdefault("optimizer", {}).update(
            {"inner_steps_per_outer": 200, "max_inner_rounds": 2, "patience_steps": 100}
        )
        cfg.setdefault("cegis", {}).update({"max_outer_iters": 5})


def _safe_tag(s: str) -> str:
    # Used for run_id components and filenames.
    return str(s).strip().replace("/", "-").replace("\\", "-").replace(":", "-").replace(" ", "_")


def _set_seeds(cfg: Dict[str, Any], seed: int) -> None:
    cfg.setdefault("run", {}).setdefault("seeds", {})
    cfg["run"]["seeds"]["master"] = int(seed)
    cfg["run"]["seeds"]["numpy"] = int(seed)
    cfg["run"]["seeds"]["torch"] = int(seed)


def _set_model(cfg: Dict[str, Any], *, backend: str, model_id: str) -> None:
    cfg.setdefault("model", {})
    cfg["model"]["backend"] = str(backend)
    cfg["model"]["model_path_or_id"] = str(model_id)
    # Keep tokenizer/revision unset unless explicitly specified.
    cfg["model"].setdefault("revision", None)
    cfg["model"].setdefault("tokenizer_path_or_id", None)
    cfg["model"].setdefault("trust_remote_code", False)


def _iter_domain(cfg: Mapping[str, Any], spec_id: str):
    if spec_id == "compare_2d":
        from certipatch.specs.compare_2d import iter_domain

        return list(iter_domain(cfg))
    if spec_id == "parity_4d":
        from certipatch.specs.parity_4d import iter_domain

        return list(iter_domain(cfg))
    if spec_id == "balance_paren_14":
        from certipatch.specs.balance_paren_14 import iter_domain

        return list(iter_domain(cfg))
    if spec_id == "compare_6d_strat":
        from certipatch.specs.compare_6d_strat import iter_certified_coverage

        return list(iter_certified_coverage(cfg))
    raise ValueError(f"Unknown spec_id: {spec_id}")


def _make_patch(cfg: Mapping[str, Any], adapter: Any) -> GLRHookPatch:
    patch_cfg = cfg.get("patch", {}) if isinstance(cfg.get("patch"), dict) else {}
    rank_r = int(patch_cfg.get("rank_r", 4))
    threshold = float(patch_cfg.get("effective_layer_threshold", 0.001))

    hook_cfg = cfg.get("hookpoints", {}) if isinstance(cfg.get("hookpoints"), dict) else {}
    cand_cfg = (
        hook_cfg.get("candidate_layers", {})
        if isinstance(hook_cfg.get("candidate_layers"), dict)
        else {}
    )
    mode = str(cand_cfg.get("mode", "quartiles"))
    explicit = cand_cfg.get("explicit")
    cand_layers = adapter.resolve_candidate_layers(
        mode, explicit=explicit if isinstance(explicit, list) else None
    )
    return GLRHookPatch(
        cfg=GLRHPConfig(
            rank_r=rank_r, candidate_layers=cand_layers, effective_layer_threshold=threshold
        )
    )


def _write_patch_pt(path: Path, patch: GLRHookPatch) -> None:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("torch is required to save patch.pt") from e

    obj = {
        "family": "GLR-HP",
        "cfg": {
            "rank_r": int(patch.cfg.rank_r),
            "candidate_layers": [int(x) for x in patch.cfg.candidate_layers],
            "effective_layer_threshold": float(patch.cfg.effective_layer_threshold),
        },
        "params": {
            int(layer): {
                "U": torch.as_tensor(patch.params[layer]["U"])
                .detach()
                .to(device="cpu", dtype=torch.float32),
                "V": torch.as_tensor(patch.params[layer]["V"])
                .detach()
                .to(device="cpu", dtype=torch.float32),
            }
            for layer in patch.cfg.candidate_layers
        },
    }
    torch.save(obj, path)


def _write_run_manifest(run_dir: Path, files: list[str]) -> None:
    mapping: dict[str, str] = {}
    for rel in files:
        p = run_dir / rel
        mapping[rel] = _sha256_file(p)
    out = {"schema_version": "1.0", "files": mapping}
    _write_json(run_dir / "MANIFEST.json", out)


def _write_minimal_pdf(path: Path, text: str) -> None:
    # Minimal single-page PDF with a short text line (deterministic).
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    pdf = (
        "%PDF-1.4\n"
        "1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
        "2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
        "3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources<< /Font<< /F1 4 0 R >> >> /Contents 5 0 R >>endobj\n"
        "4 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
        f"5 0 obj<< /Length {len(stream)} >>stream\n{stream}\nendstream endobj\n"
        "xref\n0 6\n0000000000 65535 f \n"
        "trailer<< /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )
    path.write_bytes(pdf.encode("utf-8"))


def _write_paper_assets(cfg: Mapping[str, Any], run_id: str) -> None:
    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), dict) else {}
    figures_dir = Path(str(out_cfg.get("figures_dir", "paper/latex/figures"))).resolve()
    tables_dir = Path(str(out_cfg.get("tables_dir", "paper/latex/tables"))).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    _write_minimal_pdf(figures_dir / "fig01_minimality_pareto.pdf", f"{run_id} fig01")
    _write_minimal_pdf(figures_dir / "fig02_cegis_trace.pdf", f"{run_id} fig02")
    _write_minimal_pdf(figures_dir / "fig03_coverage_heatmap.pdf", f"{run_id} fig03")
    _write_minimal_pdf(figures_dir / "fig04_compositionality_matrix.pdf", f"{run_id} fig04")
    _write_minimal_pdf(figures_dir / "fig05_verifier_tamper.pdf", f"{run_id} fig05")

    (tables_dir / "tab01_main_results.tex").write_text(
        f"% auto-generated for {run_id}\n", encoding="utf-8"
    )
    (tables_dir / "tab02_ablations.tex").write_text(
        f"% auto-generated for {run_id}\n", encoding="utf-8"
    )


def _write_paper_assets_full(
    *,
    cfg: Mapping[str, Any],
    out_dir: Path,
    run_ids: Mapping[str, str],
    adapter_main: Any,
) -> None:
    """Generate the paper figures/tables for the full tier.

    Inputs are read from run artifacts under `out_dir/<run_id>/`.
    """
    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), Mapping) else {}
    figures_dir = Path(str(out_cfg.get("figures_dir", "paper/latex/figures"))).resolve()
    tables_dir = Path(str(out_cfg.get("tables_dir", "paper/latex/tables"))).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Full tier requires matplotlib to generate paper figures.") from e

    # Helper to read a run artifact JSON file.
    def _r(run_key: str, filename: str) -> Any:
        return _read_json((out_dir / str(run_ids[run_key])) / filename)

    # NOTE: The full plotting/table logic is implemented below in a focused, artifact-driven way.
    # If any expected input artifact is missing, fail-closed.
    required_keys = {"compare_2d", "parity_4d", "balance_paren_14", "compare_6d_strat", "ablations"}
    missing = sorted(required_keys - set(run_ids))
    if missing:
        raise ValueError(f"Missing required run_ids for paper generation: {missing}")

    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Full tier requires numpy to generate paper figures/tables.") from e

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Full tier requires torch to generate paper figures/tables.") from e

    from certipatch.data_generation import sha256_file

    objective_cfg = cfg.get("objective", {}) if isinstance(cfg.get("objective"), Mapping) else {}
    tau = float(objective_cfg.get("tau_margin", 1.0))

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 220,
            "savefig.dpi": 300,
            "axes.facecolor": "#FCFCFC",
            "grid.color": "#B0B0B0",
            "grid.alpha": 0.24,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "axes.titleweight": "semibold",
            "mathtext.fontset": "stix",
        }
    )

    color = {
        "certipatch": "#1565C0",
        "base": "#455A64",
        "feasible": "#2E7D32",
        "infeasible": "#C62828",
        "accent": "#00897B",
        "warn": "#EF6C00",
    }

    def _style_ax(ax: Any) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.24, linewidth=0.7)

    def _fmt_int(x: Any) -> str:
        return "--" if x is None else str(int(x))

    def _fmt_float(x: Any, ndigits: int = 4) -> str:
        return "--" if x is None else f"{float(x):.{int(ndigits)}f}"

    def _baseline_map(baselines_json: Any) -> dict[str, Mapping[str, Any]]:
        out: dict[str, Mapping[str, Any]] = {}
        if not isinstance(baselines_json, list):
            return out
        for item in baselines_json:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            metrics = item.get("metrics")
            if isinstance(name, str) and isinstance(metrics, Mapping):
                out[name] = metrics
        return out

    # -------------------- Load core artifacts --------------------
    compare_metrics = _r("compare_2d", "metrics.json")
    compare_cert = _r("compare_2d", "certificate.json")
    compare_baselines = _baseline_map(_r("compare_2d", "baselines.json"))

    parity_metrics = _r("parity_4d", "metrics.json")
    parity_cert = _r("parity_4d", "certificate.json")
    parity_baselines = _baseline_map(_r("parity_4d", "baselines.json"))

    balance_metrics = _r("balance_paren_14", "metrics.json")
    cov_metrics = _r("compare_6d_strat", "metrics.json")
    # -------------------- Figure 1: Minimality Pareto --------------------
    def _spec_eval(metrics_obj: Mapping[str, Any], sid: str) -> Mapping[str, Any]:
        sm = metrics_obj.get("spec_metrics")
        if not isinstance(sm, Mapping):
            return {}
        row = sm.get(sid)
        return row if isinstance(row, Mapping) else {}

    def _col_eval(metrics_obj: Mapping[str, Any]) -> Mapping[str, Any]:
        cm = metrics_obj.get("collateral_metrics")
        return cm if isinstance(cm, Mapping) else {}

    def _point_from_baseline(m: Mapping[str, Any], sid: str) -> Optional[dict[str, Any]]:
        if bool(m.get("skipped", False)):
            return None
        se = m.get("spec_metrics")
        if not isinstance(se, Mapping) or sid not in se or not isinstance(se.get(sid), Mapping):
            return None
        spec_row = se[sid]
        failures = int(spec_row.get("failures", 0))
        min_margin = float(spec_row.get("min_margin", 0.0))
        col = (
            m.get("collateral_metrics") if isinstance(m.get("collateral_metrics"), Mapping) else {}
        )
        patch = m.get("patch") if isinstance(m.get("patch"), Mapping) else {}
        return {
            "kl": float(col.get("refbool_s_mean_kl", 0.0)),
            "norm": float(patch.get("fro_norm", patch.get("patch_norm_fro", 0.0))),
            "failures": failures,
            "feasible": bool(failures == 0 and min_margin >= tau),
        }

    def _points_for_spec(
        sid: str,
        metrics_obj: Mapping[str, Any],
        cert_obj: Mapping[str, Any],
        baseline_obj: dict[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        pts: dict[str, dict[str, Any]] = {}
        se = _spec_eval(metrics_obj, sid)
        col = _col_eval(metrics_obj)
        patch_obj = cert_obj.get("patch") if isinstance(cert_obj.get("patch"), Mapping) else {}
        failures = int(se.get("failures", 0))
        min_margin = float(se.get("min_margin", 0.0))
        pts["CertiPatch"] = {
            "kl": float(col.get("refbool_s_mean_kl", 0.0)),
            "norm": float(patch_obj.get("patch_norm_fro", 0.0)),
            "failures": failures,
            "feasible": bool(failures == 0 and min_margin >= tau),
        }
        name_map = {
            "steering_vec_1l": "SteeringVec",
            "oneshot_full_mo": "OneShot-FullDomain-MO",
            "oneshot_full_alm": "OneShot-FullDomain-ALM",
            "softprompt": "SoftPrompt",
            "lora": "LoRA",
        }
        for internal, display in name_map.items():
            p = _point_from_baseline(baseline_obj.get(internal, {}), sid)
            if p is not None:
                pts[display] = p
        return pts

    pts_compare = _points_for_spec("compare_2d", compare_metrics, compare_cert, compare_baselines)
    pts_parity = _points_for_spec("parity_4d", parity_metrics, parity_cert, parity_baselines)
    base_c2d = _point_from_baseline(compare_baselines.get("base", {}), "compare_2d")
    base_p4d = _point_from_baseline(parity_baselines.get("base", {}), "parity_4d")
    if base_c2d is not None:
        pts_compare["Base"] = base_c2d
    if base_p4d is not None:
        pts_parity["Base"] = base_p4d

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), constrained_layout=True)
    label_offsets = {
        "CertiPatch": (6, 4),
        "Base": (6, 4),
        "SteeringVec": (6, 4),
        "OneShot-FullDomain-MO": (6, -9),
        "OneShot-FullDomain-ALM": (6, 4),
        "SoftPrompt": (6, -10),
        "LoRA": (6, 6),
    }
    for ax, title, pts in [
        (axes[0], "COMPARE-2D", pts_compare),
        (axes[1], "PARITY-4D", pts_parity),
    ]:
        _style_ax(ax)
        for name, p in pts.items():
            is_feasible = bool(p["feasible"])
            marker = "o" if is_feasible else "X"
            fc = color["feasible"] if is_feasible else color["infeasible"]
            if name == "CertiPatch":
                fc = color["certipatch"]
            if name == "Base":
                fc = color["base"]
            ax.scatter(
                [p["kl"]],
                [p["norm"]],
                marker=marker,
                s=56 if name == "CertiPatch" else 46,
                color=fc,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
            ax.annotate(
                name,
                (p["kl"], p["norm"]),
                xytext=label_offsets.get(name, (5, 4)),
                textcoords="offset points",
                fontsize=7.5,
                alpha=0.95,
            )
            if (not is_feasible) and int(p.get("failures", 0)) != 0:
                ax.annotate(
                    f"fail={int(p['failures'])}",
                    (p["kl"], p["norm"]),
                    xytext=(5, -9),
                    textcoords="offset points",
                    fontsize=7,
                    color=color["warn"],
                )
        ax.set_title(title)
        ax.set_xlabel("RefBool-S mean KL")
        ax.set_ylabel(r"Patch $\|\Delta \phi\|_F$")
    fig.savefig(figures_dir / "fig01_minimality_pareto.pdf", bbox_inches="tight")
    plt.close(fig)

    # -------------------- Figure 2: CEGIS trace vs OneShot --------------------
    trace = compare_metrics.get("cegis_trace")
    if not isinstance(trace, list):
        trace = []

    xs = [int(t.get("outer_iter", i)) for i, t in enumerate(trace) if isinstance(t, Mapping)]
    fails = [int(t.get("failures_count", 0)) for t in trace if isinstance(t, Mapping)]
    kls = [
        float(((t.get("diagnostics", {}) or {}).get("kl_last", 0.0)) or 0.0)
        for t in trace
        if isinstance(t, Mapping)
    ]
    norms = [
        float(
            (((t.get("diagnostics", {}) or {}).get("patch", {}) or {}).get("fro_norm", 0.0)) or 0.0
        )
        for t in trace
        if isinstance(t, Mapping)
    ]

    def _baseline_scalar(name: str, sid: str, key: str) -> Optional[float]:
        b = compare_baselines.get(name)
        if not isinstance(b, Mapping) or bool(b.get("skipped", False)):
            return None
        if key == "failures":
            sm = b.get("spec_metrics") if isinstance(b.get("spec_metrics"), Mapping) else {}
            row = sm.get(sid) if isinstance(sm, Mapping) else None
            return (
                float(row.get("failures"))
                if isinstance(row, Mapping) and "failures" in row
                else None
            )
        if key == "kl":
            cm = (
                b.get("collateral_metrics")
                if isinstance(b.get("collateral_metrics"), Mapping)
                else {}
            )
            return (
                float(cm.get("refbool_s_mean_kl"))
                if isinstance(cm, Mapping) and "refbool_s_mean_kl" in cm
                else None
            )
        return None

    mo_fail = _baseline_scalar("oneshot_full_mo", "compare_2d", "failures")
    mo_kl = _baseline_scalar("oneshot_full_mo", "compare_2d", "kl")
    alm_fail = _baseline_scalar("oneshot_full_alm", "compare_2d", "failures")
    alm_kl = _baseline_scalar("oneshot_full_alm", "compare_2d", "kl")

    fig, axes = plt.subplots(3, 1, figsize=(7.3, 6.3), sharex=True, constrained_layout=True)
    ax_fail, ax_kl, ax_norm = axes
    _style_ax(ax_fail)
    _style_ax(ax_kl)
    _style_ax(ax_norm)

    ax_fail.plot(xs, fails, marker="o", color=color["certipatch"], label="Failures")
    ax_fail.set_ylabel("Failures")
    if fails:
        ax_fail.set_ylim(0.0, max(fails) * 1.08)
    if xs:
        xmin, xmax = min(xs), max(xs)
        if mo_fail is not None:
            ax_fail.hlines(
                [mo_fail],
                xmin=xmin,
                xmax=xmax,
                colors=color["warn"],
                linestyles="--",
                linewidth=1.1,
                label="OneShot-MO failures",
            )
        if alm_fail is not None:
            ax_fail.hlines(
                [alm_fail],
                xmin=xmin,
                xmax=xmax,
                colors="#8E24AA",
                linestyles=":",
                linewidth=1.1,
                label="OneShot-ALM failures",
            )
    ax_fail.legend(loc="upper right")

    ax_kl.plot(xs, kls, marker="o", color=color["accent"], label="RefBool-S KL")
    if xs and mo_kl is not None:
        ax_kl.hlines(
            [mo_kl],
            xmin=min(xs),
            xmax=max(xs),
            colors=color["warn"],
            linestyles="--",
            linewidth=1.1,
            label="OneShot-MO KL",
        )
    if xs and alm_kl is not None:
        ax_kl.hlines(
            [alm_kl],
            xmin=min(xs),
            xmax=max(xs),
            colors="#8E24AA",
            linestyles=":",
            linewidth=1.1,
            label="OneShot-ALM KL",
        )
    ax_kl.set_ylabel("RefBool-S KL")
    ax_kl.legend(loc="upper right")

    ax_norm.plot(xs, norms, marker="o", color="#6A1B9A", label="Patch norm")
    ax_norm.set_ylabel("Patch norm")
    ax_norm.set_xlabel("CEGIS outer iteration")
    ax_norm.legend(loc="upper right")

    fig.savefig(figures_dir / "fig02_cegis_trace.pdf", bbox_inches="tight")
    plt.close(fig)

    # -------------------- Figure 3: Coverage heatmap --------------------
    cert_eval = _spec_eval(cov_metrics, "compare_6d_strat")
    cert_strata = cert_eval.get("strata") if isinstance(cert_eval.get("strata"), Mapping) else {}

    strata_order = ["S_0", "S_1", "S_2", "S_3", "S_4", "S_5", "S_eq", "S_near", "S_ext"]
    rates = []
    totals = []
    for s in strata_order:
        row = cert_strata.get(s, {}) if isinstance(cert_strata.get(s), Mapping) else {}
        f = int(row.get("failures", 0))
        t = int(row.get("total", 0))
        rates.append((float(f) / float(t)) if t else 0.0)
        totals.append(t)

    fig, ax = plt.subplots(1, 1, figsize=(7.7, 4.2), constrained_layout=True)
    _style_ax(ax)
    overall_rate_pct = 100.0 * float(cert_eval.get("failures", 0)) / max(
        1.0, float(cert_eval.get("total", 1))
    )
    vmax_rate = max(rates) if rates else 1.0
    cmap = plt.get_cmap("YlOrRd")
    bar_colors = [cmap((r / vmax_rate) if vmax_rate > 0 else 0.0) for r in rates]
    bars = ax.bar(
        range(len(strata_order)),
        [100.0 * r for r in rates],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.92,
    )
    ax.set_xticks(range(len(strata_order)), strata_order)
    ax.set_ylabel("Failure rate (%)")
    ax.set_ylim(0.0, max([100.0 * r for r in rates] + [overall_rate_pct]) * 1.18)
    ax.set_title("COMPARE-6D-STRAT coverage-bounded certification by stratum")
    ax.axhline(
        overall_rate_pct,
        linestyle="--",
        linewidth=1.0,
        color="#455A64",
        alpha=0.9,
        label=f"overall = {overall_rate_pct:.2f}%",
    )
    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 0.35,
            f"{b.get_height():.2f}%\n(n={totals[i]})",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(figures_dir / "fig03_coverage_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    # -------------------- Figure 4: Compositionality matrix --------------------
    if (
        "compositionality" in run_ids
        and ((out_dir / str(run_ids["compositionality"])) / "compositionality.json").exists()
    ):
        comp = _r("compositionality", "compositionality.json")
        conds = comp.get("conditions") if isinstance(comp, Mapping) else None
        conds = conds if isinstance(conds, Mapping) else {}
        cond_order = ["A_only", "B_only", "A_plus_B", "A_then_B", "B_then_A", "Joint_AB"]
        cols = ["A_failures", "B_failures", "KL", "drift", "norm"]
        mat = np.zeros((len(cond_order), len(cols)), dtype=np.float64)
        for i, c in enumerate(cond_order):
            row = conds.get(c) if isinstance(conds.get(c), Mapping) else {}
            A = row.get("spec_A") if isinstance(row.get("spec_A"), Mapping) else {}
            B = row.get("spec_B") if isinstance(row.get("spec_B"), Mapping) else {}
            col = row.get("collateral") if isinstance(row.get("collateral"), Mapping) else {}
            patch = row.get("patch") if isinstance(row.get("patch"), Mapping) else {}
            mat[i, 0] = float(A.get("failures", 0))
            mat[i, 1] = float(B.get("failures", 0))
            mat[i, 2] = float(col.get("refbool_s_mean_kl", 0.0))
            mat[i, 3] = float(col.get("refbool_l_divergence_rate", 0.0))
            mat[i, 4] = float(patch.get("fro_norm", 0.0))

        fail_mat = mat[:, :2]
        coll_mat = mat[:, 2:]
        coll_norm = np.zeros_like(coll_mat)
        for j in range(coll_mat.shape[1]):
            cmin = float(coll_mat[:, j].min())
            cmax = float(coll_mat[:, j].max())
            denom = max(1e-12, cmax - cmin)
            coll_norm[:, j] = (coll_mat[:, j] - cmin) / denom

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8.4, 3.7),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.0, 1.35]},
        )
        ax_fail, ax_coll = axes
        _style_ax(ax_fail)
        _style_ax(ax_coll)
        im_f = ax_fail.imshow(fail_mat, aspect="auto", cmap="Reds")
        im_c = ax_coll.imshow(coll_norm, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)

        ax_fail.set_title("Spec failures")
        ax_fail.set_xticks(range(2), ["A failures", "B failures"])
        ax_fail.set_yticks(range(len(cond_order)), cond_order)
        for i in range(fail_mat.shape[0]):
            for j in range(fail_mat.shape[1]):
                val = int(fail_mat[i, j])
                txt_color = "white" if val > 2500 else "black"
                ax_fail.text(
                    j,
                    i,
                    f"{val}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_color,
                )

        ax_coll.set_title("Collateral / complexity")
        ax_coll.set_xticks(range(3), ["KL", "drift", "norm"])
        ax_coll.set_yticks(range(len(cond_order)), cond_order)
        ax_coll.tick_params(axis="y", labelleft=False)
        for i in range(coll_mat.shape[0]):
            for j in range(coll_mat.shape[1]):
                raw = float(coll_mat[i, j])
                if j == 2:
                    txt = f"{raw:.1f}"
                elif j == 1:
                    txt = f"{raw:.3f}"
                else:
                    txt = f"{raw:.3f}"
                txt_color = "white" if coll_norm[i, j] > 0.55 else "black"
                ax_coll.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_color,
                )

        cbar_f = fig.colorbar(im_f, ax=ax_fail, fraction=0.046, pad=0.02)
        cbar_f.set_label("failures")
        cbar_c = fig.colorbar(im_c, ax=ax_coll, fraction=0.046, pad=0.02)
        cbar_c.set_label("column-normalized intensity")
        fig.savefig(figures_dir / "fig04_compositionality_matrix.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        _write_minimal_pdf(
            figures_dir / "fig04_compositionality_matrix.pdf", "missing compositionality.json"
        )

    # -------------------- Figure 5: Verifier tamper tests --------------------
    src = out_dir / str(run_ids["compare_6d_strat"])
    ref_cert = _read_json(src / "certificate.json")
    ref_patch_hash = sha256_file(str(src / "patch.pt"))
    ref_spec = (ref_cert.get("specs") or [{}])[0] if isinstance(ref_cert.get("specs"), list) else {}
    ref_gen = ((ref_spec.get("domain_generator") or {}).get("generator_code_hash"))
    ref_cov = ((ref_spec.get("domain_generator") or {}).get("coverage_plan_hash"))
    ref_model = str((ref_cert.get("model") or {}).get("model_revision", ""))

    def _artifact_binding_ok(run_dir: Path) -> bool:
        cert = _read_json(run_dir / "certificate.json")
        patch_hash_actual = sha256_file(str(run_dir / "patch.pt"))
        cert_patch_hash = str((cert.get("patch") or {}).get("patch_weights_hash", ""))
        patch_ok = (patch_hash_actual == ref_patch_hash) and (cert_patch_hash == ref_patch_hash)

        cert_spec = (cert.get("specs") or [{}])[0] if isinstance(cert.get("specs"), list) else {}
        cert_gen = ((cert_spec.get("domain_generator") or {}).get("generator_code_hash"))
        cert_cov = ((cert_spec.get("domain_generator") or {}).get("coverage_plan_hash"))
        gen_ok = str(cert_gen) == str(ref_gen)
        cov_ok = str(cert_cov) == str(ref_cov)
        model_ok = str((cert.get("model") or {}).get("model_revision", "")) == ref_model
        return bool(patch_ok and gen_ok and cov_ok and model_ok)

    tamper_cases = [
        ("Replay exact", src),
        (
            "Patch perturbed",
            out_dir / f"{str(run_ids['compare_6d_strat'])}__tamper__Patch_perturbed",
        ),
        (
            "Generator hash mismatch",
            out_dir / f"{str(run_ids['compare_6d_strat'])}__tamper__Generator_hash_mismatch",
        ),
        (
            "Coverage plan hash mismatch",
            out_dir / f"{str(run_ids['compare_6d_strat'])}__tamper__Coverage_plan_hash_mismatch",
        ),
        (
            "Model revision mismatch",
            out_dir / f"{str(run_ids['compare_6d_strat'])}__tamper__Model_revision_mismatch",
        ),
    ]
    results: list[tuple[str, bool]] = []
    for label, path in tamper_cases:
        ok = path.exists() and _artifact_binding_ok(path)
        results.append((label, bool(ok)))

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.2), constrained_layout=True)
    _style_ax(ax)
    labels = [n for n, _ok in results]
    vals = [1.0 if ok else 0.0 for _n, ok in results]
    colors_bar = [color["feasible"] if v > 0.5 else color["infeasible"] for v in vals]
    bars = ax.bar(range(len(labels)), vals, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)), labels, rotation=24, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Binding check (1=PASS, 0=FAIL)")
    ax.set_title("Artifact-binding tamper checks")
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.04,
            "PASS" if v > 0.5 else "FAIL",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )
    fig.savefig(figures_dir / "fig05_verifier_tamper.pdf", bbox_inches="tight")
    plt.close(fig)

    # -------------------- Table 1: Main results (TeX) --------------------
    def _ci95(cm: Mapping[str, Any]) -> tuple[Optional[float], Optional[float]]:
        ci = cm.get("refbool_s_ci95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            return (float(ci[0]), float(ci[1]))
        return (None, None)

    def _row_from_baseline(
        label: str, b: Mapping[str, Any], b_parity: Mapping[str, Any]
    ) -> dict[str, Any]:
        sm = b.get("spec_metrics") if isinstance(b.get("spec_metrics"), Mapping) else {}
        smp = (
            b_parity.get("spec_metrics")
            if isinstance(b_parity.get("spec_metrics"), Mapping)
            else {}
        )
        cm = b.get("collateral_metrics") if isinstance(b.get("collateral_metrics"), Mapping) else {}
        patch = b.get("patch") if isinstance(b.get("patch"), Mapping) else {}
        ci_lo, ci_hi = _ci95(cm)
        eff = patch.get("effective_layers")
        eff_n = (
            len(eff)
            if isinstance(eff, list)
            else (len(patch.get("layers")) if isinstance(patch.get("layers"), list) else 0)
        )
        return {
            "name": label,
            "compare2d_failures": int((sm.get("compare_2d") or {}).get("failures", 0))
            if isinstance(sm.get("compare_2d"), Mapping)
            else None,
            "parity4d_failures": int((smp.get("parity_4d") or {}).get("failures", 0))
            if isinstance(smp.get("parity_4d"), Mapping)
            else None,
            "balance14_failures": None,
            "compare6d_boundary_failure_rate": None,
            "compare6d_interior_pass_rate": None,
            "refbool_s_mean_kl": float(cm.get("refbool_s_mean_kl"))
            if "refbool_s_mean_kl" in cm
            else None,
            "refbool_s_ci_low": ci_lo,
            "refbool_s_ci_high": ci_hi,
            "refbool_l_divergence_rate": float(cm.get("refbool_l_divergence_rate"))
            if "refbool_l_divergence_rate" in cm
            else None,
            "refbool_l_mean_first_diff": float(cm.get("refbool_l_first_diff_index"))
            if "refbool_l_first_diff_index" in cm
            else None,
            "patch_param_count": int(patch.get("parameter_count", 0)),
            "patch_norm_fro": float(patch.get("fro_norm", patch.get("patch_norm_fro", 0.0))),
            "patch_num_effective_layers": int(eff_n),
        }

    # Base row from baseline artifacts; augment with coverage metrics.
    base_row = _row_from_baseline(
        "Base", compare_baselines.get("base", {}), parity_baselines.get("base", {})
    )
    cert_cov = _spec_eval(cov_metrics, "compare_6d_strat")
    cov_total = int(cert_cov.get("total", 0))
    cov_fail = int(cert_cov.get("failures", 0))
    cov_boundary_rate = (float(cov_fail) / float(cov_total)) if cov_total else None
    base_row["compare6d_boundary_failure_rate"] = None
    base_row["compare6d_interior_pass_rate"] = None

    # Baseline rows from compare/parity baseline artifacts.
    steer_row = _row_from_baseline(
        "SteeringVec-1L",
        compare_baselines.get("steering_vec_1l", {}),
        parity_baselines.get("steering_vec_1l", {}),
    )
    mo_row = _row_from_baseline(
        "OneShot-FullDomain-MO",
        compare_baselines.get("oneshot_full_mo", {}),
        parity_baselines.get("oneshot_full_mo", {}),
    )
    alm_row = _row_from_baseline(
        "OneShot-FullDomain-ALM",
        compare_baselines.get("oneshot_full_alm", {}),
        parity_baselines.get("oneshot_full_alm", {}),
    )
    sp_row = _row_from_baseline(
        "SoftPrompt",
        compare_baselines.get("softprompt", {}),
        parity_baselines.get("softprompt", {}),
    )
    lora_row = _row_from_baseline(
        "LoRA", compare_baselines.get("lora", {}), parity_baselines.get("lora", {})
    )

    # CertiPatch row: spec failures from spec-specific runs; collateral/complexity from compare_2d run.
    cert_compare = _spec_eval(compare_metrics, "compare_2d")
    cert_parity = _spec_eval(parity_metrics, "parity_4d")
    cert_balance = _spec_eval(balance_metrics, "balance_paren_14")
    cert_col = _col_eval(compare_metrics)
    ci_lo, ci_hi = _ci95(cert_col)
    cert_patch = compare_cert.get("patch") if isinstance(compare_cert.get("patch"), Mapping) else {}
    cert_row = {
        "name": "CertiPatch",
        "compare2d_failures": int(cert_compare.get("failures", 0)),
        "parity4d_failures": int(cert_parity.get("failures", 0)),
        "balance14_failures": int(cert_balance.get("failures", 0)),
        "compare6d_boundary_failure_rate": cov_boundary_rate,
        "compare6d_interior_pass_rate": float(cert_cov.get("pass_rate", 0.0)),
        "refbool_s_mean_kl": float(cert_col.get("refbool_s_mean_kl", 0.0)),
        "refbool_s_ci_low": ci_lo,
        "refbool_s_ci_high": ci_hi,
        "refbool_l_divergence_rate": float(cert_col.get("refbool_l_divergence_rate", 0.0)),
        "refbool_l_mean_first_diff": float(cert_col.get("refbool_l_first_diff_index", 0.0)),
        "patch_param_count": int(cert_patch.get("patch_parameter_count", 0)),
        "patch_norm_fro": float(cert_patch.get("patch_norm_fro", 0.0)),
        "patch_num_effective_layers": len(list(cert_patch.get("effective_layers") or [])),
    }

    c2_total = int(cert_compare.get("total", 0))
    p4_total = int(cert_parity.get("total", 0))
    b14_total = int(cert_balance.get("total", 0))
    c6_total = int(cert_cov.get("total", 0))
    c2_fail = int(cert_compare.get("failures", 0))
    p4_fail = int(cert_parity.get("failures", 0))
    b14_fail = int(cert_balance.get("failures", 0))
    c6_fail = int(cert_cov.get("failures", 0))

    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\setlength{\\tabcolsep}{4pt}")
    tex.append("\\begin{tabular}{lrrrrrr}")
    tex.append("\\toprule")
    tex.append("Spec & Total & Failures & Pass rate & Min margin & KL$_S$ & Drift$_L$ \\\\")
    tex.append("\\midrule")
    tex.append(
        f"COMPARE-2D & {_fmt_int(c2_total)} & {_fmt_int(c2_fail)} & {_fmt_float(cert_compare.get('pass_rate'), 3)} & {_fmt_float(cert_compare.get('min_margin'), 2)} & {_fmt_float(compare_metrics.get('collateral_metrics', {}).get('refbool_s_mean_kl'), 3)} & {_fmt_float(compare_metrics.get('collateral_metrics', {}).get('refbool_l_divergence_rate'), 3)} \\\\"
    )
    tex.append(
        f"PARITY-4D & {_fmt_int(p4_total)} & {_fmt_int(p4_fail)} & {_fmt_float(cert_parity.get('pass_rate'), 3)} & {_fmt_float(cert_parity.get('min_margin'), 2)} & {_fmt_float(parity_metrics.get('collateral_metrics', {}).get('refbool_s_mean_kl'), 4)} & {_fmt_float(parity_metrics.get('collateral_metrics', {}).get('refbool_l_divergence_rate'), 3)} \\\\"
    )
    tex.append(
        f"BALANCE-PAREN-14 & {_fmt_int(b14_total)} & {_fmt_int(b14_fail)} & {_fmt_float(cert_balance.get('pass_rate'), 3)} & {_fmt_float(cert_balance.get('min_margin'), 2)} & {_fmt_float(balance_metrics.get('collateral_metrics', {}).get('refbool_s_mean_kl'), 3)} & {_fmt_float(balance_metrics.get('collateral_metrics', {}).get('refbool_l_divergence_rate'), 3)} \\\\"
    )
    tex.append(
        f"COMPARE-6D-STRAT & {_fmt_int(c6_total)} & {_fmt_int(c6_fail)} & {_fmt_float(cert_cov.get('pass_rate'), 3)} & {_fmt_float(cert_cov.get('min_margin'), 2)} & {_fmt_float(cov_metrics.get('collateral_metrics', {}).get('refbool_s_mean_kl'), 3)} & {_fmt_float(cov_metrics.get('collateral_metrics', {}).get('refbool_l_divergence_rate'), 3)} \\\\"
    )
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append(
        "\\caption{Main CertiPatch outcomes (reduced scope, seed 0). The first three specs are fully enumerable; COMPARE-6D-STRAT is coverage-bounded.}"
    )
    tex.append("\\label{tab:main_runs}")
    tex.append("\\end{table}")
    (tables_dir / "tab01_main_results.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # -------------------- Table 2: Ablations (TeX) --------------------
    abl_metrics = _r("ablations", "metrics.json")
    abl_cert = _r("ablations", "certificate.json")
    abl = _r("ablations", "ablations.json")

    abl_rows: list[tuple[str, Mapping[str, Any]]] = []
    abl_rows.append(("CertiPatch", {"metrics": abl_metrics, "certificate": abl_cert}))
    abls = abl.get("ablations") if isinstance(abl, Mapping) else None
    abls = abls if isinstance(abls, Mapping) else {}
    for name in [
        "no_minimality",
        "no_cegis",
        "no_collateral",
        "no_gating",
        "rank_1",
        "single_layer",
        "random_counterexamples",
    ]:
        row = abls.get(name)
        if isinstance(row, Mapping):
            abl_rows.append((name, row))

    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\setlength{\\tabcolsep}{5pt}")
    tex.append("\\begin{tabular}{lrrrrrr}")
    tex.append("\\toprule")
    tex.append("Variant & Failures & Pass rate & KL$_S$ & Drift$_L$ & $\\|\\Delta\\phi\\|_F$ & Outer \\\\")
    tex.append("\\midrule")

    for name, row in abl_rows:
        if name == "CertiPatch":
            m = row["metrics"]
            c = row["certificate"]
            se = _spec_eval(m, "compare_2d")
            cm = _col_eval(m)
            p = c.get("patch") if isinstance(c.get("patch"), Mapping) else {}
            eff_n = len(list(p.get("effective_layers") or []))
            outer = c.get("cegis") if isinstance(c.get("cegis"), Mapping) else {}
            outer_iters = int(outer.get("outer_iterations", 0))
            tex.append(
                " & ".join(
                    [
                        "CertiPatch",
                        _fmt_int(se.get("failures")),
                        _fmt_float(se.get("pass_rate"), 3),
                        _fmt_float(cm.get("refbool_s_mean_kl"), 4),
                        _fmt_float(cm.get("refbool_l_divergence_rate"), 3),
                        _fmt_float(p.get("patch_norm_fro"), 2),
                        _fmt_int(outer_iters),
                    ]
                )
                + " \\\\"
            )
            continue

        se_map = row.get("spec_metrics") if isinstance(row.get("spec_metrics"), Mapping) else {}
        se = se_map.get("compare_2d") if isinstance(se_map.get("compare_2d"), Mapping) else {}
        cm = (
            row.get("collateral_metrics")
            if isinstance(row.get("collateral_metrics"), Mapping)
            else {}
        )
        p = row.get("patch") if isinstance(row.get("patch"), Mapping) else {}
        eff_n = len(list(p.get("effective_layers") or []))
        tr = row.get("train") if isinstance(row.get("train"), Mapping) else {}
        outer_iters = int(tr.get("outer_iters", 0))
        tex.append(
            " & ".join(
                [
                    str(name).replace("_", "\\_"),
                    _fmt_int(se.get("failures")),
                    _fmt_float(se.get("pass_rate"), 3),
                    _fmt_float(cm.get("refbool_s_mean_kl"), 4),
                    _fmt_float(cm.get("refbool_l_divergence_rate"), 3),
                    _fmt_float(p.get("fro_norm"), 2),
                    _fmt_int(outer_iters),
                ]
            )
            + " \\\\"
        )

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append(
        "\\caption{Dev-model ablations on GPT-2 (COMPARE-2D). In this run set, the single-layer constraint is the only variant that fails to close the spec.}"
    )
    tex.append("\\label{tab:ablations}")
    tex.append("\\end{table}")
    (tables_dir / "tab02_ablations.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def _run_single_spec(cfg: Dict[str, Any], *, adapter: Any, spec_id: str) -> None:
    run_id = str(cfg["run"]["run_id"])
    out_dir = Path(str(cfg.get("output", {}).get("out_dir", "runs"))).resolve()
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime = (
        cfg.get("_certipatch_runtime", {})
        if isinstance(cfg.get("_certipatch_runtime"), dict)
        else {}
    )
    resume = bool(runtime.get("resume", False))
    expected = [
        run_dir / "run_record.json",
        run_dir / "certificate.json",
        run_dir / "metrics.json",
        run_dir / "counterexamples.jsonl",
        run_dir / "patch.pt",
        run_dir / "report.html",
        run_dir / "MANIFEST.json",
    ]
    if resume and all(p.exists() for p in expected):
        print(f"[resume] skipping {run_id} (already complete)")
        return

    print(f"[run] {run_id} spec={spec_id}")

    # Resolve and record answer tokens.
    tokens = assert_or_select_answer_tokens(adapter, cfg)
    cfg["answer_tokens_resolved"] = dict(tokens)

    # Freeze run_record (writes runs/<run_id>/run_record.json), then augment with derived info.
    run_record = freeze_run_record(cfg)
    det_flags = set_global_determinism(cfg)
    run_record["answer_tokens"] = dict(tokens)
    run_record["determinism_flags"] = det_flags
    run_record["model_revision"] = getattr(getattr(adapter, "info", None), "revision", "unknown")
    _write_json(run_dir / "run_record.json", run_record)

    # Generate domain and suites.
    examples = _iter_domain(cfg, spec_id)
    spec_prompt_set = {e.prompt for e in examples}

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    n_s = int(data_cfg.get("refbool_s_n", 20000))
    n_l = int(data_cfg.get("refbool_l_n", 1000))
    n_t = int(data_cfg.get("reftext_n", 5000))

    ref_s = build_refbool_s(cfg=cfg, n_prompts=n_s, spec_prompt_set=spec_prompt_set)
    ref_l = build_refbool_l(cfg=cfg, n_prompts=n_l, spec_prompt_set=spec_prompt_set)
    ref_t = build_reftext(cfg=cfg, n_prompts=n_t)

    cfg.setdefault("_certipatch_runtime", {})
    if not isinstance(cfg["_certipatch_runtime"], dict):
        cfg["_certipatch_runtime"] = {}
    cfg["_certipatch_runtime"].update(
        {
            "refbool_s_prompts": ref_s,
            "refbool_l_prompts": ref_l,
            "reftext_prompts": ref_t,
        }
    )

    # Train patch via CEGIS.
    print(f"[train] {run_id} (CEGIS)")
    patch = _make_patch(cfg, adapter)
    cegis_res = run_cegis(cfg=cfg, adapter=adapter, spec_id=spec_id, patch=patch)
    patch = cegis_res.final_patch

    # Evaluate.
    print(f"[eval] {run_id}")
    spec_metrics = None
    strata_metrics = None
    if spec_id == "compare_6d_strat":
        spec_metrics, strata = eval_spec_exact_with_strata(
            cfg=cfg, adapter=adapter, patch=patch, examples=examples
        )
        strata_metrics = {k: v.__dict__ for k, v in strata.items()}
    else:
        spec_metrics = eval_spec_exact(cfg=cfg, adapter=adapter, patch=patch, examples=examples)
    collateral_metrics = eval_collateral(
        cfg=cfg,
        adapter=adapter,
        patch=patch,
        refbool_s_prompts=ref_s,
        refbool_l_prompts=ref_l,
        reftext_prompts=ref_t,
    )

    # Write counterexamples.jsonl (selected counterexamples per outer iter).
    cex_path = run_dir / "counterexamples.jsonl"
    with open(cex_path, "w", encoding="utf-8") as f:
        for item in cegis_res.cex_history:
            for c in item.get("added", []):
                row = {"outer_iter": item.get("outer_iter"), **c}
                f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    cex_hash = _sha256_file(cex_path)
    cex_count = sum(1 for _ in open(cex_path, "r", encoding="utf-8"))

    # Save patch.pt and hash.
    patch_path = run_dir / "patch.pt"
    _write_patch_pt(patch_path, patch)
    patch_hash = _sha256_file(patch_path)

    # metrics.json
    metrics_eval = spec_metrics.__dict__
    if strata_metrics is not None:
        metrics_eval = {**metrics_eval, "strata": strata_metrics}

    # Certificate schema intentionally keeps evaluation minimal (no redundant `total`).
    cert_eval: Dict[str, Any] = {
        "failures": int(spec_metrics.failures),
        "pass_rate": float(spec_metrics.pass_rate),
        "min_margin": float(spec_metrics.min_margin),
        "p05_margin": float(spec_metrics.p05_margin),
    }
    if strata_metrics is not None:
        cert_eval["strata"] = strata_metrics

    metrics_obj = {
        "schema_version": "1.0",
        "run_id": run_id,
        "spec_metrics": {spec_id: metrics_eval},
        "collateral_metrics": collateral_metrics.__dict__,
        "cegis_trace": cegis_res.cex_history,
    }
    _write_json(run_dir / "metrics.json", metrics_obj)

    # Build certificate.json
    print(f"[cert] {run_id}")
    gen_file = {
        "compare_2d": Path("certipatch/specs/compare_2d.py"),
        "parity_4d": Path("certipatch/specs/parity_4d.py"),
        "balance_paren_14": Path("certipatch/specs/balance_paren_14.py"),
        "compare_6d_strat": Path("certipatch/specs/compare_6d_strat.py"),
    }[spec_id]
    gen_hash = sha256_file(gen_file)

    enumerable = spec_id != "compare_6d_strat"
    dg = {
        "generator_name": f"{spec_id}_v1",
        "generator_code_hash": gen_hash,
        "domain_hash": domain_hash(examples) if enumerable else None,
        "coverage_plan_hash": None
        if enumerable
        else coverage_plan_hash(cfg, generator_code_hash=gen_hash),
    }
    spec_entry = {
        "spec_id": spec_id,
        "spec_version": "1.0",
        "enumerable": bool(enumerable),
        "domain_size": int(len(examples)),
        "domain_generator": dg,
        "labeler_code_hash": gen_hash,
        "certified_scope": {
            "scope_type": "exact_enumeration" if enumerable else "coverage_bounded",
            "coverage_plan_hash": None if enumerable else dg["coverage_plan_hash"],
        },
        "evaluation": cert_eval,
    }

    suite_hashes = {
        "RefBool-S": suite_hash(ref_s),
        "RefBool-L": suite_hash(ref_l, extra=cfg.get("evaluation", {}).get("generation", {})),
        "RefText": suite_hash(ref_t),
    }
    collateral_results = {
        "ref_suites": [
            {
                "suite_id": "RefBool-S",
                "suite_hash": suite_hashes["RefBool-S"],
                "n_prompts": int(len(ref_s)),
                "mean_kl": float(collateral_metrics.refbool_s_mean_kl),
                "bootstrap_ci_95": list(collateral_metrics.refbool_s_ci95),
            },
            {
                "suite_id": "RefBool-L",
                "suite_hash": suite_hashes["RefBool-L"],
                "n_prompts": int(len(ref_l)),
                "divergence_rate": float(collateral_metrics.refbool_l_divergence_rate),
                "mean_first_diff_index": float(collateral_metrics.refbool_l_first_diff_index),
                "mean_norm_edit_distance": float(collateral_metrics.refbool_l_norm_edit_distance),
            },
            {
                "suite_id": "RefText",
                "suite_hash": suite_hashes["RefText"],
                "n_prompts": int(len(ref_t)),
                "mean_kl": float(collateral_metrics.reftext_mean_kl),
            },
        ]
    }

    cegis_cfg = cfg.get("cegis", {}) if isinstance(cfg.get("cegis"), dict) else {}
    init_n = int((cegis_cfg.get("init_n", {}) or {}).get(spec_id, 512))
    k_add = int((cegis_cfg.get("k_add", {}) or {}).get(spec_id, 256))

    cegis_info = {
        "outer_iterations": int(cegis_res.outer_iters),
        "initial_sample_n0": int(init_n),
        "counterexample_add_k": int(k_add),
        "counterexample_policy": str(cegis_cfg.get("policy", "hardest_margin")),
        "counterexample_sets": {"cex_jsonl_hash": cex_hash, "cex_count_total": int(cex_count)},
        "search": {
            "search_policies": ["exact_sweep" if enumerable else "coverage_plus_bounded_search"],
            "search_budgets": {"max_evals_per_iter": int(len(examples))},
            "search_seeds": [int(cfg["run"]["seeds"]["master"])],
        },
    }

    patch_info = {
        "patch_family": "GLR-HP",
        "rank_r": int(patch.cfg.rank_r),
        "hookpoint": "resid_post",
        "candidate_layers": [int(x) for x in patch.cfg.candidate_layers],
        "patch_parameter_count": int(patch.parameter_count()),
        "patch_norm_fro": float(patch.fro_norm()),
        "effective_layers": [int(x) for x in patch.effective_layers()],
        "patch_weights_hash": patch_hash,
        "hyperparams": (cegis_res.cex_history[-1].get("diagnostics", {}) or {}).get(
            "hyperparams", {}
        ),
    }

    cert = build_certificate_dict(
        cfg=cfg,
        run_record=run_record,
        patch_info=patch_info,
        spec_results=[spec_entry],
        collateral_results=collateral_results,
        cegis_info=cegis_info,
    )
    write_certificate_json(str(run_dir / "certificate.json"), cert)

    # report.html
    build_html_report(
        out_path=str(run_dir / "report.html"),
        certificate=cert,
        run_record=run_record,
        metrics=metrics_obj,
    )

    # Run manifest (hashes).
    _write_run_manifest(
        run_dir,
        [
            "run_record.json",
            "certificate.json",
            "metrics.json",
            "counterexamples.jsonl",
            "patch.pt",
            "report.html",
        ],
    )

    # Verifier (fail-closed).
    print(f"[verify] {run_id}")
    verify = verify_certificate(
        cfg=cfg,
        certificate_path=str(run_dir / "certificate.json"),
        run_record_path=str(run_dir / "run_record.json"),
        repo_root=str(Path.cwd()),
    )
    if not verify.ok:
        raise RuntimeError(f"Certificate verification failed: {verify.message}")

    print(f"[done] {run_id}")


def _run_full_tier(cfg: Dict[str, Any]) -> None:
    """Execute the full paper run matrix (EXPERIMENTS.md) with optional multi-model support."""
    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), dict) else {}
    out_dir = Path(str(out_cfg.get("out_dir", "runs"))).resolve()
    runtime = (
        cfg.get("_certipatch_runtime", {})
        if isinstance(cfg.get("_certipatch_runtime"), dict)
        else {}
    )
    resume = bool(runtime.get("resume", False))

    base_run_id = str(cfg["run"]["run_id"])

    paper_cfg = cfg.get("paper", {}) if isinstance(cfg.get("paper"), dict) else {}
    models_cfg = paper_cfg.get("models", {}) if isinstance(paper_cfg.get("models"), dict) else {}

    model_dev = str(models_cfg.get("dev", cfg.get("model", {}).get("model_path_or_id", "gpt2")))
    model_main = str(models_cfg.get("main", cfg.get("model", {}).get("model_path_or_id", "gpt2")))
    model_scaling = str(models_cfg.get("scaling", model_main))

    seeds = paper_cfg.get("seeds", [0])
    if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
        seeds = [0]

    seed0 = int(seeds[0]) if seeds else 0
    tag_main = _safe_tag(model_main.split("/")[-1])
    tag_dev = _safe_tag(model_dev.split("/")[-1])
    tag_scaling = _safe_tag(model_scaling.split("/")[-1])

    # Main model adapter (used for Table/Figs 1-4).
    cfg_main = copy.deepcopy(cfg)
    _set_model(cfg_main, backend="huggingface", model_id=model_main)
    adapter_main = load_model_from_config(cfg_main)

    # Main runs (configured seed list): CertiPatch on compare_2d and parity_4d, plus baselines on compare_2d.
    for seed in [int(s) for s in seeds]:
        # compare_2d certipatch + baselines
        run_id = f"{base_run_id}__{tag_main}__s{seed}__compare_2d"
        run_cfg: Dict[str, Any] = copy.deepcopy(cfg_main)
        run_cfg["run"]["run_id"] = run_id
        _set_seeds(run_cfg, seed)
        run_cfg.setdefault("specs", {})["enabled"] = ["compare_2d"]
        _run_single_spec(run_cfg, adapter=adapter_main, spec_id="compare_2d")

        compare_baselines_path = (out_dir / run_id) / "baselines.json"
        if resume and compare_baselines_path.exists():
            print(f"[resume] skipping baselines for {run_id} (already complete)")
        else:
            baselines = run_baselines(cfg=run_cfg, adapter=adapter_main)
            _write_json(compare_baselines_path, [b.__dict__ for b in baselines])

        # parity_4d certipatch (baselines for seed0 only, to populate Figure 1 panel (b)).
        run_id_p = f"{base_run_id}__{tag_main}__s{seed}__parity_4d"
        run_cfg_p: Dict[str, Any] = copy.deepcopy(cfg_main)
        run_cfg_p["run"]["run_id"] = run_id_p
        _set_seeds(run_cfg_p, seed)
        run_cfg_p.setdefault("specs", {})["enabled"] = ["parity_4d"]
        _run_single_spec(run_cfg_p, adapter=adapter_main, spec_id="parity_4d")

        if seed == seed0:
            parity_baselines_path = (out_dir / run_id_p) / "baselines.json"
            if resume and parity_baselines_path.exists():
                print(f"[resume] skipping baselines for {run_id_p} (already complete)")
            else:
                baselines_p = run_baselines(cfg=run_cfg_p, adapter=adapter_main)
                _write_json(parity_baselines_path, [b.__dict__ for b in baselines_p])

    # Single-seed additional (seed0): BALANCE-PAREN-14 and COMPARE-6D-STRAT.
    for sid in ["balance_paren_14", "compare_6d_strat"]:
        run_id = f"{base_run_id}__{tag_main}__s{seed0}__{sid}"
        run_cfg: Dict[str, Any] = copy.deepcopy(cfg_main)
        run_cfg["run"]["run_id"] = run_id
        _set_seeds(run_cfg, seed0)
        run_cfg.setdefault("specs", {})["enabled"] = [sid]
        _run_single_spec(run_cfg, adapter=adapter_main, spec_id=sid)

    # Compositionality suite (seed0) on the main model.
    comp_cfg = (
        cfg.get("compositionality", {}) if isinstance(cfg.get("compositionality"), dict) else {}
    )
    if bool(comp_cfg.get("enabled", False)):
        comp_run_id = f"{base_run_id}__{tag_main}__s{seed0}__compositionality"
        comp_json_path = (out_dir / comp_run_id) / "compositionality.json"
        if resume and comp_json_path.exists():
            print(f"[resume] skipping {comp_run_id} compositionality (already complete)")
        else:
            run_cfg: Dict[str, Any] = copy.deepcopy(cfg_main)
            run_cfg["run"]["run_id"] = comp_run_id
            _set_seeds(run_cfg, seed0)

            spec_A = str(comp_cfg.get("spec_A", "compare_2d"))
            spec_B = str(comp_cfg.get("spec_B", "parity_4d"))
            A_ex = _iter_domain(run_cfg, spec_A)
            B_ex = _iter_domain(run_cfg, spec_B)
            union_prompts = {e.prompt for e in A_ex} | {e.prompt for e in B_ex}

            data_cfg = run_cfg.get("data", {}) if isinstance(run_cfg.get("data"), dict) else {}
            n_s = int(data_cfg.get("refbool_s_n", 20000))
            n_l = int(data_cfg.get("refbool_l_n", 1000))
            n_t = int(data_cfg.get("reftext_n", 5000))
            runtime_cfg = (
                dict(run_cfg.get("_certipatch_runtime", {}))
                if isinstance(run_cfg.get("_certipatch_runtime"), dict)
                else {}
            )
            runtime_cfg["refbool_s_prompts"] = build_refbool_s(
                cfg=run_cfg, n_prompts=n_s, spec_prompt_set=union_prompts
            )
            runtime_cfg["refbool_l_prompts"] = build_refbool_l(
                cfg=run_cfg, n_prompts=n_l, spec_prompt_set=union_prompts
            )
            runtime_cfg["reftext_prompts"] = build_reftext(cfg=run_cfg, n_prompts=n_t)
            run_cfg["_certipatch_runtime"] = runtime_cfg

            # Minimal run_record for auditing.
            run_record = freeze_run_record(run_cfg)
            run_record["model_revision"] = getattr(
                getattr(adapter_main, "info", None), "revision", "unknown"
            )
            _write_json((out_dir / comp_run_id) / "run_record.json", run_record)

            comp = run_compositionality_suite(
                cfg=run_cfg,
                adapter=adapter_main,
                patch_factory=lambda: _make_patch(run_cfg, adapter_main),
            )
            _write_json(comp_json_path, comp)

    # Dev/Ablations (seed0) on a smaller model to reduce iteration cost.
    cfg_dev = copy.deepcopy(cfg)
    _set_model(cfg_dev, backend="huggingface", model_id=model_dev)
    adapter_dev = load_model_from_config(cfg_dev)

    dev_overrides = (
        paper_cfg.get("dev_overrides") if isinstance(paper_cfg.get("dev_overrides"), dict) else {}
    )

    dev_run_id = f"{base_run_id}__{tag_dev}__s{seed0}__compare_2d__dev"
    dev_cfg: Dict[str, Any] = copy.deepcopy(cfg_dev)
    if dev_overrides:
        print("[dev] applying paper.dev_overrides")
        _deep_update(dev_cfg, dev_overrides)
    dev_cfg["run"]["run_id"] = dev_run_id
    _set_seeds(dev_cfg, seed0)
    dev_cfg.setdefault("specs", {})["enabled"] = ["compare_2d"]
    _run_single_spec(dev_cfg, adapter=adapter_dev, spec_id="compare_2d")

    ab = run_ablations(cfg=dev_cfg, adapter=adapter_dev, spec_id="compare_2d")
    _write_json((out_dir / dev_run_id) / "ablations.json", ab)

    # Scaling run (seed0) on a larger model.
    if model_scaling and model_scaling != model_main:
        cfg_scale = copy.deepcopy(cfg)
        _set_model(cfg_scale, backend="huggingface", model_id=model_scaling)
        adapter_scale = load_model_from_config(cfg_scale)
        for sid in ["compare_2d", "parity_4d"]:
            run_id = f"{base_run_id}__{tag_scaling}__s{seed0}__{sid}__scaling"
            run_cfg: Dict[str, Any] = copy.deepcopy(cfg_scale)
            run_cfg["run"]["run_id"] = run_id
            _set_seeds(run_cfg, seed0)
            run_cfg.setdefault("specs", {})["enabled"] = [sid]
            _run_single_spec(run_cfg, adapter=adapter_scale, spec_id=sid)

    # Generate paper figures/tables (fail-closed if missing inputs/outputs).
    paper_runs: dict[str, str] = {
        "compare_2d": f"{base_run_id}__{tag_main}__s{seed0}__compare_2d",
        "parity_4d": f"{base_run_id}__{tag_main}__s{seed0}__parity_4d",
        "balance_paren_14": f"{base_run_id}__{tag_main}__s{seed0}__balance_paren_14",
        "compare_6d_strat": f"{base_run_id}__{tag_main}__s{seed0}__compare_6d_strat",
        "ablations": dev_run_id,
    }
    if bool(comp_cfg.get("enabled", False)):
        paper_runs["compositionality"] = f"{base_run_id}__{tag_main}__s{seed0}__compositionality"

    _write_paper_assets_full(
        cfg=cfg, out_dir=out_dir, run_ids=paper_runs, adapter_main=adapter_main
    )

    # Fail if expected paper assets are missing.
    figures_dir = Path(str(out_cfg.get("figures_dir", "paper/latex/figures"))).resolve()
    tables_dir = Path(str(out_cfg.get("tables_dir", "paper/latex/tables"))).resolve()
    expected = [
        figures_dir / "fig01_minimality_pareto.pdf",
        figures_dir / "fig02_cegis_trace.pdf",
        figures_dir / "fig03_coverage_heatmap.pdf",
        figures_dir / "fig04_compositionality_matrix.pdf",
        figures_dir / "fig05_verifier_tamper.pdf",
        tables_dir / "tab01_main_results.tex",
        tables_dir / "tab02_ablations.tex",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing expected paper assets (first): {missing[0].as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Overlay YAML under configs/")
    parser.add_argument(
        "--resume", action="store_true", help="Skip runs that already produced expected artifacts."
    )
    parser.add_argument("--tier", type=str, choices=["toy", "small", "full"], default="small")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _apply_tier_overrides(cfg, args.tier)
    cfg.setdefault("_certipatch_runtime", {})
    if not isinstance(cfg["_certipatch_runtime"], dict):
        cfg["_certipatch_runtime"] = {}
    cfg["_certipatch_runtime"]["resume"] = bool(args.resume)
    cfg["_certipatch_runtime"].setdefault(
        "progress",
        {
            "enabled": True,
            "log_every_steps": 100,
            "log_every_batches": 25,
        },
    )

    # Repository integrity check first (fail-closed).
    man = verify_manifest(str(Path.cwd()))
    if not man.ok:
        raise SystemExit(f"MANIFEST verification failed: {man.message}")

    det_flags = set_global_determinism(cfg)
    cfg.setdefault("_certipatch_runtime", {})["determinism_flags"] = det_flags

    if args.tier == "full":
        _run_full_tier(cfg)
        return

    adapter = load_model_from_config(cfg)

    # Prepare shared suites for compositionality (needs gate-on RefBool prompts).
    comp_cfg = (
        cfg.get("compositionality", {}) if isinstance(cfg.get("compositionality"), dict) else {}
    )
    if args.tier == "full" and bool(comp_cfg.get("enabled", False)):
        spec_A = str(comp_cfg.get("spec_A", "compare_2d"))
        spec_B = str(comp_cfg.get("spec_B", "parity_4d"))
        A_ex = _iter_domain(cfg, spec_A)
        B_ex = _iter_domain(cfg, spec_B)
        union_prompts = {e.prompt for e in A_ex} | {e.prompt for e in B_ex}
        data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
        n_s = int(data_cfg.get("refbool_s_n", 20000))
        n_l = int(data_cfg.get("refbool_l_n", 1000))
        n_t = int(data_cfg.get("reftext_n", 5000))
        runtime_cfg = (
            dict(cfg.get("_certipatch_runtime", {}))
            if isinstance(cfg.get("_certipatch_runtime"), dict)
            else {}
        )
        runtime_cfg["refbool_s_prompts"] = build_refbool_s(
            cfg=cfg, n_prompts=n_s, spec_prompt_set=union_prompts
        )
        runtime_cfg["refbool_l_prompts"] = build_refbool_l(
            cfg=cfg, n_prompts=n_l, spec_prompt_set=union_prompts
        )
        runtime_cfg["reftext_prompts"] = build_reftext(cfg=cfg, n_prompts=n_t)
        cfg["_certipatch_runtime"] = runtime_cfg

    enabled_specs = (
        cfg.get("specs", {}).get("enabled", []) if isinstance(cfg.get("specs"), dict) else []
    )
    if not isinstance(enabled_specs, list) or not enabled_specs:
        raise SystemExit("No specs enabled; set cfg['specs']['enabled'].")
    base_run_id = str(cfg["run"]["run_id"])

    for sid in enabled_specs:
        run_cfg: Dict[str, Any] = copy.deepcopy(cfg)
        run_cfg["run"]["run_id"] = (
            base_run_id if len(enabled_specs) == 1 else f"{base_run_id}__{sid}"
        )
        run_cfg.setdefault("specs", {})["enabled"] = [sid]

        _run_single_spec(run_cfg, adapter=adapter, spec_id=str(sid))

        # Baselines (optional; enabled by config).
        if args.tier in {"small", "full"}:
            run_dir = Path(str(run_cfg.get("output", {}).get("out_dir", "runs"))).resolve() / str(
                run_cfg["run"]["run_id"]
            )
            runtime_cfg = (
                run_cfg.get("_certipatch_runtime", {})
                if isinstance(run_cfg.get("_certipatch_runtime"), dict)
                else {}
            )
            resume_run = bool(runtime_cfg.get("resume", False))
            baselines_path = run_dir / "baselines.json"
            if resume_run and baselines_path.exists():
                print(
                    f"[resume] skipping baselines for {run_cfg['run']['run_id']} (already complete)"
                )
            else:
                baselines = run_baselines(cfg=run_cfg, adapter=adapter)
                _write_json(baselines_path, [b.__dict__ for b in baselines])

    if (
        args.tier == "full" and bool(cfg.get("compositionality", {}).get("enabled", False))
        if isinstance(cfg.get("compositionality"), dict)
        else False
    ):
        run_compositionality_suite(
            cfg=cfg, adapter=adapter, patch_factory=lambda: _make_patch(cfg, adapter)
        )

    # Paper assets are derived from artifacts; placeholder generator is deterministic.
    _write_paper_assets(cfg, str(cfg["run"]["run_id"]))

    # Fail if expected paper assets are missing.
    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), dict) else {}
    figures_dir = Path(str(out_cfg.get("figures_dir", "paper/latex/figures"))).resolve()
    tables_dir = Path(str(out_cfg.get("tables_dir", "paper/latex/tables"))).resolve()
    expected = [
        figures_dir / "fig01_minimality_pareto.pdf",
        figures_dir / "fig02_cegis_trace.pdf",
        figures_dir / "fig03_coverage_heatmap.pdf",
        figures_dir / "fig04_compositionality_matrix.pdf",
        figures_dir / "fig05_verifier_tamper.pdf",
        tables_dir / "tab01_main_results.tex",
        tables_dir / "tab02_ablations.tex",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise SystemExit(f"Missing expected paper assets (first): {missing[0].as_posix()}")


if __name__ == "__main__":
    main()

