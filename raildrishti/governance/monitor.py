"""Model health, SLA tracking, latency and resource monitoring."""
from __future__ import annotations

import time

import numpy as np


# Which stored metric maps to which SLA key, and the direction of "good".
SLA_RULES = {
    "min_auc": ("auc", ">="),
    "min_accuracy": ("accuracy", ">="),
    "min_recall": ("recall", ">="),
    "min_f1": ("f1", ">="),
    "min_ndcg_at_5": ("auc", ">="),      # ranker proxy
    "max_fpr": ("fpr", "<="),
    "max_mae_min": ("mae", "<="),
    "max_mape": ("mape", "<="),
    "max_hallucination_rate": ("hallucination_rate", "<="),
    "min_groundedness": ("groundedness", ">="),
    "min_task_success": ("task_success", ">="),
    "max_latency_ms": ("latency_ms", "<="),
}


def evaluate_sla(sla: dict, metrics: dict) -> dict:
    checks = []
    for key, threshold in sla.items():
        if key not in SLA_RULES:
            continue
        metric, op = SLA_RULES[key]
        if metric not in metrics or metrics[metric] is None:
            continue
        val = metrics[metric]
        ok = (val >= threshold) if op == ">=" else (val <= threshold)
        checks.append({"sla": key, "metric": metric, "value": round(float(val), 4),
                       "threshold": threshold, "op": op, "ok": bool(ok)})
    breached = [c for c in checks if not c["ok"]]
    return {
        "checks": checks,
        "breached": breached,
        "status": "healthy" if not breached else ("degraded" if len(breached) == 1 else "critical"),
        "slas_met": len(checks) - len(breached),
        "slas_total": len(checks),
    }


class LatencyMonitor:
    """Rolling latency stats per model (p50/p95) against the SLA budget."""

    def __init__(self):
        self.samples: dict[str, list[float]] = {}

    def time_call(self, model: str, fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        dt = (time.perf_counter() - t0) * 1000.0
        self.samples.setdefault(model, []).append(dt)
        return out

    def record(self, model: str, ms: float):
        self.samples.setdefault(model, []).append(ms)

    def stats(self, model: str) -> dict:
        s = self.samples.get(model, [])
        if not s:
            return {"n": 0, "p50": None, "p95": None, "mean": None}
        return {"n": len(s), "p50": round(float(np.percentile(s, 50)), 2),
                "p95": round(float(np.percentile(s, 95)), 2),
                "mean": round(float(np.mean(s)), 2)}

    def resource_utilization(self) -> dict:
        """Proxy service-health / resource signal (CPU/mem via psutil if present)."""
        out = {"cpu_pct": None, "mem_pct": None, "available": False}
        try:
            import psutil
            out.update({"cpu_pct": round(psutil.cpu_percent(interval=None), 1),
                        "mem_pct": round(psutil.virtual_memory().percent, 1),
                        "available": True})
        except Exception:
            pass
        return out
