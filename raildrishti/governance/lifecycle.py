"""Automated model lifecycle management.

Retraining-trigger logic, challenger training, a validation gate (champion vs
challenger on the SLA), versioned registration, promotion and rollback.
"""
from __future__ import annotations

import time

import pandas as pd
from sklearn.model_selection import train_test_split

from .audit import AuditTrail
from .monitor import evaluate_sla
from .registry import ModelRegistry


class LifecycleManager:
    def __init__(self, registry: ModelRegistry, audit: AuditTrail):
        self.registry = registry
        self.audit = audit
        self._version_counter: dict[str, int] = {}

    # -- trigger logic ------------------------------------------------------
    @staticmethod
    def should_retrain(drift_report: dict, perf_drop: float = 0.0,
                       n_new_samples: int = 0, min_samples: int = 2000) -> dict:
        reasons = []
        if drift_report.get("severity") == "drift":
            reasons.append(f"data drift (max PSI {drift_report.get('max_psi')}, "
                           f"{drift_report.get('n_drifted')} features drifted)")
        if perf_drop > 0.05:
            reasons.append(f"concept drift / performance drop {perf_drop:.3f}")
        if n_new_samples >= min_samples:
            reasons.append(f"{n_new_samples} new labelled samples available")
        # Retrain when drift OR performance drop is present; new data alone only triggers
        # if it is substantial.
        trigger = (drift_report.get("severity") == "drift") or perf_drop > 0.05 or n_new_samples >= 4 * min_samples
        return {"trigger": bool(trigger), "reasons": reasons}

    # -- challenger training -------------------------------------------------
    def retrain(self, name: str, builder, df: pd.DataFrame, reason: str = "scheduled",
                test_size: float = 0.25, seed: int = 7) -> dict:
        model = builder()
        strat = df["is_fraud"] if name == "fraud_detector" else None
        train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
        model.fit(train)
        metrics = model.evaluate(test)

        incumbent = self.registry.get(name)
        incumbent_metrics = incumbent["metrics"]
        challenger_sla = evaluate_sla(model.spec.sla, metrics)
        incumbent_sla = evaluate_sla(model.spec.sla, incumbent_metrics)

        improved = self._is_better(name, metrics, incumbent_metrics)
        acceptable = self._acceptable(name, metrics, incumbent_metrics)
        gate = {
            "challenger_meets_sla": challenger_sla["status"] != "critical",
            "improved_on_incumbent": improved,
            "not_worse_than_incumbent": acceptable,
            "incumbent_status": incumbent_sla["status"],
        }
        # A challenger is promoted if it meets SLA and is at least as good as the
        # incumbent within tolerance (retraining refreshes the model on recent data).
        promote = (challenger_sla["status"] in ("healthy", "degraded") and acceptable)

        n = self._version_counter.get(name, 0) + 1
        self._version_counter[name] = n
        version = f"v{n + 1}-auto"
        meta = dict(incumbent["meta"])
        meta.update({"n_train": len(train), "n_test": len(test),
                     "retrain_reason": reason, "feature_importance": model.feature_importance()})
        self.registry.register(name, model, metrics, meta, version=version, stage="staging")
        self.audit.record("retrain", actor="lifecycle_manager",
                          payload={"model": name, "version": version, "reason": reason,
                                   "metrics": metrics, "gate": gate})

        result = {"model": name, "challenger_version": version, "metrics": metrics,
                  "gate": gate, "promoted": False}
        if promote:
            self.registry.promote(name, version)
            self.audit.record("promote", actor="lifecycle_manager",
                              payload={"model": name, "version": version, "reason": "challenger accepted"})
            result["promoted"] = True
        else:
            self.audit.record("reject_challenger", actor="lifecycle_manager",
                              payload={"model": name, "version": version, "gate": gate},
                              severity="warning")
        return result

    @staticmethod
    def _is_better(name: str, new: dict, old: dict) -> bool:
        if name in ("delay_predictor", "fare_predictor"):
            return new.get("mae", 1e9) < old.get("mae", 1e9)
        if name == "fraud_detector":
            return new.get("f1", 0) > old.get("f1", 0)
        return new.get("auc", 0) > old.get("auc", 0)

    @staticmethod
    def _acceptable(name: str, new: dict, old: dict, tol: float = 0.01) -> bool:
        """Challenger is no worse than incumbent beyond a small tolerance."""
        if name in ("delay_predictor", "fare_predictor"):
            return new.get("mae", 1e9) <= old.get("mae", 1e9) * (1 + tol)
        if name == "fraud_detector":
            return new.get("f1", 0) >= old.get("f1", 0) - tol
        return new.get("auc", 0) >= old.get("auc", 0) - tol

    # -- rollback ------------------------------------------------------------
    def rollback(self, name: str, reason: str) -> dict:
        out = self.registry.rollback(name)
        self.audit.record("rollback", actor="lifecycle_manager",
                          payload={"model": name, "reason": reason, **out}, severity="warning")
        return out
