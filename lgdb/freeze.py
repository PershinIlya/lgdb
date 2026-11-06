"""Freeze: hash prompts, code, and config so later changes are detectable."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def _files(root: Path) -> list[Path]:
    return sorted(list((root / "prompts").glob("*.txt")) + list((root / "lgdb").glob("*.py"))
                  + [root / "config.yaml"])


def digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in _files(root)}


def freeze(root: Path, models: dict) -> dict:
    rec = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "models": models,
           "files": digest(root)}
    (root / "freeze.json").write_text(json.dumps(rec, indent=1))
    return rec


def check(root: Path) -> list[str]:
    """Files changed since freezing; empty list means the pipeline is still frozen."""
    path = root / "freeze.json"
    if not path.exists():
        return ["freeze.json missing"]
    frozen = json.loads(path.read_text())["files"]
    now = digest(root)
    return sorted({k for k in set(frozen) | set(now) if frozen.get(k) != now.get(k)})
