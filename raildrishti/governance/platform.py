"""The Centralized AI Governance Platform (module 8).

A single orchestrator that holds the registry, audit trail, monitors and
lifecycle manager, and exposes every governance capability the brief asks for:
model-health/SLA tracking, data-quality gates, drift + concept-drift detection,
bias/hallucination monitoring, explainability, root-cause analysis, automated
lifecycle actions, and an audit trail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DATA_DIR, GOV_DIR, MODEL_MANIFEST
from ..models.predictors import ALL_BUILDERS, DATA_FILES
from .audit import AuditTrail
from .bias import evaluate_bias, hallucination_report
from .data_quality import validate
from .drift import detect_behavior_change, detect_concept_drift, detect_data_drift
from .explain import explain_in_words, explain_prediction, global_importance
from .lifecycle import LifecycleManager
from .monitor import LatencyMonitor, evaluate_sla
from .registry import ModelRegistry
from . import rca as rca_mod


class GovernancePlatform:
    def __init__(self, registry: ModelRegistry | None = None, audit: AuditTrail | None = None):
        self.registry = registry or ModelRegistry()
        self.audit = audit or AuditTrail()
        self.latency = LatencyMonitor()
        self.lifecycle = LifecycleManager(self.registry, self.audit)
        self.reference: dict[str, pd.DataFrame] = {}
        self.current: dict[str, pd.DataFrame] = {}
        self._model_cache: dict[str, object] = {}

    # -- models --------------------------------------------------------------
    def get_model(self, name: str):
        key = f"{name}:{self.registry.live_version(name)}"
        if key not in self._model_cache:
            self._model_cache[key] = self.registry.load(name)
        return self._model_cache[key]

    def _reference(self, name: str, n: int = 4000) -> pd.DataFrame:
        if name not in DATA_FILES:
            return pd.DataFrame()
        if name not in self.reference:
            df = pd.read_csv(DATA_DIR / f"{DATA_FILES[name]}.csv")
            self.reference[name] = df.sample(min(n, len(df)), random_state=0).reset_index(drop=True)
        return self.reference[name]

    # -- model health / SLA --------------------------------------------------
    def health(self, name: str) -> dict:
        entry = self.registry.get(name)
        metrics = dict(entry["metrics"])
        lat = self.latency.stats(name)
        if lat["p95"] is not None:
            metrics["latency_ms"] = lat["p95"]
        sla = evaluate_sla(entry["meta"].get("sla", MODEL_MANIFEST.get(name, {}).get("sla", {})), metrics)
        return {
            "model": name, "live_version": entry["version"], "stage": entry["stage"],
            "metrics": metrics, "sla": sla, "latency": lat,
            "service_health": "up" if sla["status"] != "critical" else "degraded",
        }

    def health_all(self) -> dict:
        out = {}
        for name in self.registry.list_models():
            out[name] = self.health(name)
        return out

    # -- data quality --------------------------------------------------------
    def data_quality(self, name: str, df: pd.DataFrame, key: str | None = None) -> dict:
        schema = {}
        spec = self._spec(name)
        if spec:
            for c in spec.numeric_features:
                if c in df.columns:
                    schema[c] = {"dtype": "numeric"}
        report = validate(df, schema=schema, key=key)
        if not report["passed"]:
            self.audit.record("data_quality_fail", actor="governance_platform",
                              payload={"model": name, "quality_score": report["quality_score"],
                                       "issues": report["issues"]}, severity="warning")
        return report

    def _spec(self, name: str):
        model = self.get_model(name) if self.registry.live_version(name) else None
        return getattr(model, "spec", None)

    # -- drift ---------------------------------------------------------------
    def drift(self, name: str, current: pd.DataFrame | None = None) -> dict:
        if name not in DATA_FILES:
            # Generative / agentic models have no tabular feature reference here.
            return {"features": {}, "n_drifted": 0, "max_psi": 0.0, "severity": "stable"}
        ref = self._reference(name)
        cur = current if current is not None else self.current.get(name, ref)
        spec = self._spec(name)
        feats = spec.drift_features if spec else list(ref.columns)
        report = detect_data_drift(ref, cur, [f for f in feats if f in ref.columns and f in cur.columns])
        if report["severity"] != "stable":
            self.audit.record("drift_detected", actor="governance_platform",
                              payload={"model": name, **{k: report[k] for k in ("severity", "n_drifted", "max_psi")}},
                              severity="warning" if report["severity"] == "watch" else "high")
        return report

    def concept_drift(self, name: str, perf_history: list[float], baseline: float) -> dict:
        return detect_concept_drift(perf_history, baseline)

    # -- bias ----------------------------------------------------------------
    def bias(self, name: str, df: pd.DataFrame | None = None) -> list[dict]:
        spec = self._spec(name)
        if not spec or not spec.protected_attrs:
            return []
        df = df if df is not None else self._reference(name)
        model = self.get_model(name)
        out = []
        for attr in spec.protected_attrs:
            if attr not in df.columns:
                continue
            if spec.task in ("classification", "anomaly_classification"):
                y_pred = model.predict(df)
                y_true = df[spec.target].values if spec.target in df.columns else y_pred
                res = evaluate_bias(y_true, y_pred, df[attr].astype(str).rename(attr))
            else:  # regression: error parity across groups
                pred = model.predict(df)
                y = df[spec.target].values
                err = np.abs(y - pred)
                g = df[attr].astype(str)
                groups = {str(k): round(float(v), 3) for k, v in err.groupby(g).mean().items()}
                if len(groups) >= 2:
                    spread = max(groups.values()) - min(groups.values())
                    res = {"attribute": attr, "metric": "mean_abs_error",
                           "groups": groups, "spread": round(spread, 3),
                           "flagged": spread > 0.25 * (np.mean(list(groups.values())) + 1e-6),
                           "severity": "medium" if spread > 0.25 * (np.mean(list(groups.values())) + 1e-6) else "low"}
                else:
                    continue
            out.append(res)
            if res.get("flagged"):
                self.audit.record("bias_flagged", actor="governance_platform",
                                  payload={"model": name, "attribute": attr, "result": res},
                                  severity="warning")
        return out

    # -- explainability ------------------------------------------------------
    def explain(self, name: str, row: pd.DataFrame) -> dict:
        model = self.get_model(name)
        model.pipeline._governed_model = model
        res = explain_prediction(model.pipeline, model.X(row), governed_model=model)
        res["narrative"] = explain_in_words(name, res.get("top_features", []))
        return res


    def global_importance(self, name: str) -> list[dict]:
        model = self.get_model(name)
        return global_importance(model.pipeline.named_steps["model"], model.spec.features)

    # -- root cause analysis -------------------------------------------------
    def root_cause(self, name: str, current: pd.DataFrame | None = None,
                   performance_drop: float = 0.0, degraded_models: set[str] | None = None,
                   feed_status: dict | None = None) -> dict:
        drift_report = self.drift(name, current)
        model = self.get_model(name)
        imp = model.feature_importance()
        result = rca_mod.analyze(name, drift_report, imp, performance_drop,
                                 degraded_models, feed_status)
        self.audit.record("rca", actor="governance_platform",
                          payload={"model": name, "origin": result["origin"],
                                   "top_cause": (result["ranked_causes"][0]["feature"]
                                                 if result["ranked_causes"] else None)})
        return result

    # -- lifecycle actions ---------------------------------------------------
    def retrain(self, name: str, df: pd.DataFrame | None = None, reason: str = "manual") -> dict:
        builder = ALL_BUILDERS[name]
        df = df if df is not None else pd.read_csv(DATA_DIR / f"{DATA_FILES[name]}.csv")
        result = self.lifecycle.retrain(name, builder, df, reason=reason)
        self._model_cache = {k: v for k, v in self._model_cache.items() if not k.startswith(name + ":")}
        return result

    def rollback(self, name: str, reason: str = "manual") -> dict:
        out = self.lifecycle.rollback(name, reason)
        self._model_cache = {k: v for k, v in self._model_cache.items() if not k.startswith(name + ":")}
        return out

    # -- drift simulation (demo / stress-test helper) ------------------------
    def inject_drift(self, name: str, feature: str, magnitude: float = 0.5,
                     frac: float = 0.6, seed: int = 1) -> pd.DataFrame:
        """Produce a 'current' window whose `feature` distribution has shifted —
        used to exercise the closed governance loop end-to-end."""
        rng = np.random.default_rng(seed)
        cur = (self.current.get(name).copy() if name in self.current
               else self._reference(name).copy())
        n = len(cur)
        k = int(frac * n)
        idx = rng.choice(n, k, replace=False)
        cur[feature] = cur[feature].astype(float)
        col = cur[feature]
        span = float(col.max() - col.min()) or 1.0
        cur.loc[cur.index[idx], feature] = col.iloc[idx] + magnitude * span * rng.uniform(0.5, 1.5, k)
        self.current[name] = cur
        self.audit.record("drift_injected", actor="demo_harness",
                          payload={"model": name, "feature": feature, "magnitude": magnitude},
                          severity="warning")
        return cur

    # -- governance report ---------------------------------------------------
    def report(self) -> dict:
        health = self.health_all()
        ok, n_events = self.audit.verify_chain()
        statuses = [h["sla"]["status"] for h in health.values()]
        return {
            "platform": "RailDrishti AI Governance",
            "models_governed": len(health),
            "healthy": statuses.count("healthy"),
            "degraded": statuses.count("degraded"),
            "critical": statuses.count("critical"),
            "overall": "critical" if "critical" in statuses else
                       ("degraded" if "degraded" in statuses else "healthy"),
            "audit_chain_intact": ok,
            "audit_events": n_events,
            "resource": self.latency.resource_utilization(),
            "models": health,
        }
