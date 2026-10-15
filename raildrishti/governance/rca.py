"""Autonomous root-cause analysis.

Given a degraded model, correlates which drifted features actually matter (using
feature importance), then walks the platform dependency graph to decide whether
the cause is local (this model's data) or inherited (an upstream model/feed).
Emits ranked causes and recommended remediations.
"""
from __future__ import annotations

# Which upstream feeds / models each governed model depends on.
DEPENDENCIES = {
    "confirmation_predictor": ["bookings_feed"],
    "delay_predictor": ["train_runs_feed", "weather_feed", "station_congestion_feed"],
    "fare_predictor": ["fares_feed", "demand_feed"],
    "recommender": ["search_sessions_feed", "delay_predictor", "fare_predictor"],
    "fraud_detector": ["users_feed"],
    "support_assistant": ["kb_feed"],
    "travel_planner_agent": ["recommender", "delay_predictor", "fare_predictor", "confirmation_predictor"],
}


def rank_causes(drift_report: dict, feature_importance: dict, performance_drop: float) -> list[dict]:
    """Rank candidate causes by (drift severity x feature importance)."""
    causes = []
    for feat, info in drift_report.get("features", {}).items():
        imp = float(feature_importance.get(feat, 0.0))
        sev = {"stable": 0.1, "watch": 0.5, "drift": 1.0}[info["severity"]]
        score = sev * (0.3 + imp)
        if info["severity"] != "stable" and imp > 0:
            causes.append({
                "feature": feat,
                "psi": info["psi"],
                "severity": info["severity"],
                "importance": round(imp, 4),
                "ref_mean": info.get("ref_mean"),
                "cur_mean": info.get("cur_mean"),
                "cause_score": round(score, 4),
            })
    causes.sort(key=lambda c: c["cause_score"], reverse=True)
    return causes


def trace_dependencies(model: str, degraded_models: set[str]) -> list[str]:
    """Return the upstream models (in dependency order) that could be the source."""
    chain = []
    for dep in DEPENDENCIES.get(model, []):
        if dep in degraded_models:
            chain.append(dep)
    return chain


def analyze(model: str, drift_report: dict, feature_importance: dict,
            performance_drop: float, degraded_models: set[str] | None = None,
            feed_status: dict | None = None) -> dict:
    degraded_models = degraded_models or set()
    feed_status = feed_status or {}
    causes = rank_causes(drift_report, feature_importance, performance_drop)
    upstream = trace_dependencies(model, degraded_models)

    origin = "local"
    explanation = []
    if upstream:
        origin = "inherited"
        explanation.append(f"Upstream model(s) degraded: {', '.join(upstream)}")
    if causes:
        top = causes[0]
        explanation.append(
            f"Feature '{top['feature']}' drifted (PSI={top['psi']}, {top['severity']}) and carries "
            f"importance {top['importance']}; mean moved {top.get('ref_mean')} -> {top.get('cur_mean')}.")
    bad_feeds = [f for f, ok in feed_status.items() if not ok]
    if bad_feeds:
        origin = "upstream_feed"
        explanation.append(f"Data feed(s) failing quality gate: {', '.join(bad_feeds)}")

    if origin == "inherited":
        remediation = ["Investigate upstream model degradation first",
                       "Re-validate downstream model after upstream is restored",
                       "Replay recent traffic against restored upstream version"]
    elif origin == "upstream_feed":
        remediation = ["Quarantine the failing feed and apply input validation",
                       "Backfill from the last known-good snapshot",
                       "Retrain once clean data volume threshold is met"]
    else:
        remediation = ["Trigger automated retraining on the latest window",
                       "Recompute feature baselines for drift monitoring",
                       "Validate challenger on hold-out before promotion"]

    return {
        "model": model,
        "origin": origin,
        "performance_drop": round(performance_drop, 4),
        "ranked_causes": causes[:5],
        "upstream_models": upstream,
        "explanation": " ".join(explanation) if explanation else "No significant cause detected.",
        "recommended_remediations": remediation,
        "confidence": round(min(0.5 + 0.1 * len(causes) + 0.2 * len(upstream), 0.99), 3),
    }
