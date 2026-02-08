from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from certipatch.artifacts.verifier import verify_manifest


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def test_verify_manifest_roundtrip(tmp_path: Path) -> None:
    a_path = tmp_path / "a.txt"
    a_bytes = b"hello\n"
    a_path.write_bytes(a_bytes)
    a_hash = _sha256_bytes(a_bytes)

    # Self-hash rule: compute SHA over manifest text with the manifest line hash replaced by 64 zeros.
    zero = "0" * 64
    manifest_zero = (f"{a_hash}  a.txt\n{zero}  MANIFEST.sha256\n").encode("utf-8")
    self_hash = _sha256_bytes(manifest_zero)

    manifest_final = (f"{a_hash}  a.txt\n{self_hash}  MANIFEST.sha256\n").encode("utf-8")

    (tmp_path / "MANIFEST.sha256").write_bytes(manifest_final)

    res = verify_manifest(str(tmp_path), "MANIFEST.sha256")
    assert res.ok, res.message


def test_verify_manifest_detects_hash_mismatch(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"hello\n")
    (tmp_path / "MANIFEST.sha256").write_bytes(b"0" * 64 + b"  MANIFEST.sha256\n")

    res = verify_manifest(str(tmp_path), "MANIFEST.sha256")
    assert not res.ok
