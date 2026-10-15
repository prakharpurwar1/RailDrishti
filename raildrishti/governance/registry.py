"""Model registry with versioning, promotion and rollback.

Keeps a catalogue of every governed model: its versions, the metrics each was
trained to, which version is live in each environment, and the SLA it must meet.
Backed by a JSON catalogue plus joblib artifact files on disk.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import joblib

from ..config import MODEL_DIR, GOV_DIR


class ModelRegistry:
    def __init__(self, model_dir: Path = MODEL_DIR, catalogue: Path | None = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.catalogue_path = Path(catalogue or (GOV_DIR / "model_registry.json"))
        self.catalogue_path.parent.mkdir(parents=True, exist_ok=True)
        self._cat = self._load()

    def _load(self) -> dict:
        if self.catalogue_path.exists():
            return json.loads(self.catalogue_path.read_text())
        return {"models": {}}

    def _save(self):
        self.catalogue_path.write_text(json.dumps(self._cat, indent=2, default=str))

    # -- registration -------------------------------------------------------
    def register(self, name: str, artifact, metrics: dict, meta: dict,
                 version: str | None = None, stage: str = "staging") -> dict:
        version = version or f"v{int(time.time())}"
        rec = self._cat["models"].setdefault(name, {"versions": {}, "live": None, "history": []})
        # Store the FILENAME only (portable): the artifact is always resolved
        # against this machine's MODEL_DIR at load time, never an absolute path.
        filename = f"{name}__{version}.joblib"
        path = self.model_dir / filename
        joblib.dump(artifact, path)
        rec["versions"][version] = {
            "version": version,
            "filename": filename,
            "path": filename,
            "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
            "meta": meta,
            "stage": stage,
            "registered_at": time.time(),
        }
        rec["history"].append({"action": "register", "version": version, "ts": time.time()})
        # First registration goes live automatically.
        if rec["live"] is None:
            rec["live"] = version
            rec["versions"][version]["stage"] = "production"
        self._save()
        return rec["versions"][version]

    # -- access -------------------------------------------------------------
    def get(self, name: str, version: str | None = None):
        rec = self._cat["models"].get(name)
        if not rec:
            raise KeyError(f"model '{name}' not registered")
        version = version or rec["live"]
        return rec["versions"][version]

    def _artifact_path(self, rec: dict) -> Path:
        """Resolve a version's artifact to a local file.

        Handles both new records (portable filename) and legacy records that
        stored an absolute path from another machine (e.g. a different checkout).
        """
        raw = rec.get("filename") or rec.get("path") or ""
        p = Path(raw)
        if p.is_absolute() and p.exists():
            return p                       # path already valid on this machine
        return self.model_dir / p.name     # resolve by filename against current MODEL_DIR

    def load(self, name: str, version: str | None = None):
        rec = self.get(name, version)
        path = self._artifact_path(rec)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact '{path.name}' not found in {self.model_dir}. "
                f"Rebuild artifacts with:  python scripts/train_all.py  &&  "
                f"python scripts/evaluate_agents.py")
        return joblib.load(path)

    def live_version(self, name: str) -> str | None:
        return self._cat["models"].get(name, {}).get("live")

    def versions(self, name: str) -> list[str]:
        return list(self._cat["models"].get(name, {}).get("versions", {}).keys())

    def list_models(self) -> list[str]:
        return list(self._cat["models"].keys())

    # -- lifecycle ----------------------------------------------------------
    def promote(self, name: str, version: str, to_stage: str = "production") -> dict:
        rec = self._cat["models"][name]
        if version not in rec["versions"]:
            raise KeyError(version)
        prev = rec["live"]
        rec["live"] = version
        rec["versions"][version]["stage"] = to_stage
        if prev and prev in rec["versions"]:
            rec["versions"][prev]["stage"] = "archived"
        rec["history"].append({"action": "promote", "version": version, "from": prev, "ts": time.time()})
        self._save()
        return {"promoted": version, "previous": prev}

    def rollback(self, name: str) -> dict:
        """Revert to the most recent archived version that is not live."""
        rec = self._cat["models"][name]
        live = rec["live"]
        archived = [v for v, d in rec["versions"].items()
                    if v != live and d.get("stage") in ("archived", "staging")]
        if not archived:
            raise RuntimeError(f"no rollback target for {name}")
        target = sorted(archived, key=lambda v: rec["versions"][v]["registered_at"])[-1]
        rec["live"] = target
        rec["versions"][target]["stage"] = "production"
        if live:
            rec["versions"][live]["stage"] = "rolled_back"
        rec["history"].append({"action": "rollback", "version": target, "from": live, "ts": time.time()})
        self._save()
        return {"rolled_back_to": target, "from": live}

    def update_metrics(self, name: str, version: str, metrics: dict):
        self._cat["models"][name]["versions"][version]["metrics"].update(
            {k: float(v) for k, v in metrics.items()})
        self._save()

    def summary(self) -> dict:
        out = {}
        for name, rec in self._cat["models"].items():
            live = rec["live"]
            out[name] = {
                "live_version": live,
                "stage": rec["versions"][live]["stage"] if live else None,
                "metrics": rec["versions"][live]["metrics"] if live else {},
                "n_versions": len(rec["versions"]),
                "meta": rec["versions"][live]["meta"] if live else {},
            }
        return out
