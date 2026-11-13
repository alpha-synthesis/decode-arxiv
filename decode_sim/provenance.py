from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

MANIFESTS = Path(__file__).resolve().parents[1] / "manifests"


def semantic_sha256(config_path) -> str:
    cfg = yaml.safe_load(Path(config_path).read_text())
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def config_matches(recorded_sha, config_path) -> tuple[bool, str]:
    got = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    if recorded_sha in (None, got):
        return True, got
    sem = semantic_sha256(config_path)
    for m in MANIFESTS.glob("*.json"):
        d = json.loads(m.read_text())
        if d.get("released_config_sha256") == recorded_sha and d.get("config_semantic_sha256") == sem:
            return True, got
    return False, got
