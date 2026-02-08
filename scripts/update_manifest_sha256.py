"""scripts.update_manifest_sha256

Regenerate the repository-level MANIFEST.sha256.

Run from repo root:
  python scripts/update_manifest_sha256.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from certipatch.artifacts.verifier import verify_manifest, write_repo_manifest

    write_repo_manifest(str(repo_root))
    res = verify_manifest(str(repo_root))
    if not res.ok:
        raise SystemExit(f"MANIFEST regeneration failed verification: {res.message}")
    print(res.message)


if __name__ == "__main__":
    main()
