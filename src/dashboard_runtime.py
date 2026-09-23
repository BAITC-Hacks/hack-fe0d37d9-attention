"""Explicit dashboard action; delegates every forecast/audit decision to the agent.

The read-only model adapter resolves the existing versioned layout locally so
artifacts copied from another machine do not depend on its absolute paths.
Manifest files on disk, fitted models, hashes, and forecasting policy are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib

from .agent import ForecastOrchestrator
from .config import ARTIFACTS_DIR, DATA_CACHE_DIR
from .weather import OpenMeteoSingleRunsClient


def load_local_models(root: Path) -> tuple[dict, dict]:
    root = Path(root).resolve()
    manifest = json.loads((root / "models/final/manifest.json").read_text(encoding="utf-8"))
    version = manifest["model_version"]
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", version):
        raise ValueError("Invalid frozen model version")
    if set(manifest["files"]) != {"T1", "T2"}:
        raise ValueError("Frozen manifest must cover both turbines")
    paths = {}
    # Check all hashes before deserializing any model, using only local versioned
    # files. Never fall back to an external path stored in the old manifest.
    for turbine, record in manifest["files"].items():
        path = (root / "models/final" / version / f"{turbine}.joblib").resolve()
        if not path.is_relative_to(root / "models/final"):
            raise ValueError("Model is outside the repository artifact directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Frozen model hash mismatch")
        paths[turbine] = path
    models = {turbine: joblib.load(path) for turbine, path in paths.items()}
    for turbine, path in paths.items():
        manifest["files"][turbine]["path"] = str(path)
    return manifest, models


def run_cached_agent(origin: str, *, with_ai: bool = False, root: Path = ARTIFACTS_DIR,
                     cache_dir: Path = DATA_CACHE_DIR) -> str:
    manifest, models = load_local_models(root)
    agent = ForecastOrchestrator(client=OpenMeteoSingleRunsClient(cache_dir=cache_dir, cache_only=True),
                                 manifest=manifest, models=models, root=root)
    result = agent.run(origin, no_llm=not with_ai)
    return result.context.run_id
