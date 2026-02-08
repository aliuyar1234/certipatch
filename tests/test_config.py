from __future__ import annotations

from pathlib import Path

import pytest

from certipatch.config import load_config, validate_config


def test_load_config_deep_merge_and_list_replace(tmp_path: Path) -> None:
    default_yaml = tmp_path / "default.yaml"
    overlay_yaml = tmp_path / "overlay.yaml"

    default_yaml.write_text(
        "\n".join(
            [
                "run:",
                "  run_id: auto",
                "  device: cpu",
                "  dtype: float32",
                "  seeds:",
                "    master: 0",
                "    numpy: 0",
                "    torch: 0",
                "",
                "model:",
                "  backend: huggingface",
                "  model_path_or_id: gpt2",
                "  revision: null",
                "  tokenizer_path_or_id: null",
                "  trust_remote_code: false",
                "",
                "answer_tokens:",
                '  primary: {"yes": " Yes", "no": " No"}',
                '  fallback: {"yes": " true", "no": " false"}',
                "",
                "hookpoints:",
                "  kind: resid_post",
                "  apply_at: answer_position_last_nonpad",
                "  candidate_layers:",
                "    mode: explicit",
                "    explicit: [0, 1]",
                "",
                "patch:",
                "  family: GLR-HP",
                "  rank_r: 4",
                "",
                "specs:",
                '  enabled: ["compare_2d"]',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    overlay_yaml.write_text(
        "\n".join(
            [
                "run:",
                "  run_id: test_run",
                "  seeds:",
                "    torch: 7",
                "specs:",
                '  enabled: ["parity_4d"]',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(str(overlay_yaml), default_path=str(default_yaml))
    assert cfg["run"]["run_id"] == "test_run"
    assert cfg["run"]["seeds"]["master"] == 0
    assert cfg["run"]["seeds"]["numpy"] == 0
    assert cfg["run"]["seeds"]["torch"] == 7
    assert cfg["specs"]["enabled"] == ["parity_4d"]


def test_validate_config_error_message_has_pointer() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_config({})
    # Required-property errors are usually rooted at "/".
    assert "Config validation failed at" in str(excinfo.value)
