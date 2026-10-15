"""Explainability: per-prediction attributions and global importance (SHAP)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def explain_prediction(model, X_row: pd.DataFrame, background: pd.DataFrame | None = None,
                       top_k: int = 5, governed_model: Any = None) -> dict:
    """Return the top contributing features for a single prediction."""
    governed = governed_model or (model if hasattr(model, "feature_importance") else getattr(model, "_governed_model", None))
    pipe = getattr(model, "pipeline", model)
    try:
        import shap
        try:
            expl = shap.TreeExplainer(pipe)
            vals = expl.shap_values(X_row)
            if isinstance(vals, list):          # multi-class -> take positive class
                vals = vals[-1]
            vals = np.asarray(vals)
            if vals.ndim == 3:
                vals = vals[0, :, -1]
            else:
                vals = vals.reshape(-1)[: X_row.shape[1]]
        except Exception:
            expl = shap.Explainer(pipe, background if background is not None else X_row)
            sv = expl(X_row)
            vals = np.asarray(sv.values)
            vals = vals.reshape(-1)[: X_row.shape[1]]
        feats = list(X_row.columns)
        contrib = sorted(zip(feats, [float(v) for v in vals]), key=lambda t: abs(t[1]), reverse=True)
        return {
            "method": "shap",
            "top_features": [{"feature": f, "contribution": round(v, 4)} for f, v in contrib[:top_k]],
        }
    except Exception as e:  # graceful fallback
        if governed is not None and hasattr(governed, "feature_importance"):
            imp = governed.feature_importance()
            pairs = list(imp.items())
            return {
                "method": "feature_importance_fallback",
                "top_features": [{"feature": f, "contribution": round(float(v), 4)} for f, v in pairs[:top_k]],
            }
        importances = getattr(pipe, "feature_importances_", None)
        if importances is None and hasattr(pipe, "named_steps") and "model" in pipe.named_steps:
            importances = getattr(pipe.named_steps["model"], "feature_importances_", None)
        if importances is None:
            return {"method": "unavailable", "error": str(e), "top_features": []}
        feats = list(X_row.columns)
        pairs = sorted(zip(feats, importances), key=lambda t: abs(t[1]), reverse=True)
        return {"method": "feature_importance_fallback",
                "top_features": [{"feature": f, "contribution": round(float(v), 4)} for f, v in pairs[:top_k]]}



def global_importance(model, feature_names: list[str], top_k: int = 10) -> list[dict]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return []
    pairs = sorted(zip(feature_names, imp), key=lambda t: t[1], reverse=True)
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs[:top_k]]


def explain_in_words(model_name: str, top_features: list[dict]) -> str:
    """Turn attributions into a human-readable rationale for the passenger/agent."""
    if not top_features:
        return f"{model_name}: no explanation available."
    parts = []
    for t in top_features[:3]:
        direction = "increases" if t["contribution"] >= 0 else "reduces"
        parts.append(f"{t['feature']} {direction} the score (impact {t['contribution']:+.3f})")
    return f"{model_name} decision driven mainly by: " + "; ".join(parts) + "."
