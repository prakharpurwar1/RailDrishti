"""Tamper-evident audit trail.

Every governance action (registration, prediction batch, drift alert, retrain,
promotion, rollback, agent decision) is appended as one JSON line, chained with
the hash of the previous record so tampering is detectable. This satisfies the
"maintain complete audit trails" / "auditability" requirement of the brief.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from ..config import AUDIT_LOG


class AuditTrail:
    def __init__(self, path: Path = AUDIT_LOG):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._tail_hash()

    def _tail_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        last = None
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
        if not last:
            return "0" * 64
        try:
            return json.loads(last)["hash"]
        except Exception:
            return "0" * 64

    def record(self, event_type: str, actor: str, payload: dict, severity: str = "info") -> dict:
        with self._lock:
            self._last_hash = self._tail_hash()
            entry = {
                "ts": time.time(),
                "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "event_type": event_type,
                "actor": actor,
                "severity": severity,
                "payload": payload,
                "prev_hash": self._last_hash,
            }
            body = json.dumps(entry, sort_keys=True, default=str)
            entry["hash"] = hashlib.sha256((self._last_hash + body).encode()).hexdigest()
            self._last_hash = entry["hash"]
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            return entry

    def read(self, limit: int | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows[-limit:] if limit else rows

    def verify_chain(self) -> tuple[bool, int]:
        """Recompute the hash chain; return (is_intact, n_records)."""
        prev = "0" * 64
        n = 0
        for rec in self.read():
            body = json.dumps({k: rec[k] for k in rec if k != "hash"}, sort_keys=True, default=str)
            expect = hashlib.sha256((prev + body).encode()).hexdigest()
            if expect != rec.get("hash"):
                return False, n
            prev = rec["hash"]
            n += 1
        return True, n
