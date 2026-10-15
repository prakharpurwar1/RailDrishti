"""Governed model base: every RailDrishti model is a first-class, instrumented
citizen of the governance platform. It carries its feature spec, produces its
own evaluation metrics, and can explain its predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelSpec:
    name: str
    task: str                       # classification | regression | ranking | anomaly | generation | agentic
    target: str | None
    numeric_features: list[str]
    categorical_features: list[str] = field(default_factory=list)
    protected_attrs: list[str] = field(default_factory=list)
    drift_features: list[str] = field(default_factory=list)
    sla: dict = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return self.numeric_features + self.categorical_features


class GovernedModel:
    """Wraps an sklearn estimator with the platform's feature pipeline + metadata."""

    def __init__(self, spec: ModelSpec, estimator, threshold: float = 0.5):
        self.spec = spec
        self.estimator = estimator
        self.threshold = threshold
        self.pipeline = self._build_pipeline(estimator)

    def _build_pipeline(self, estimator) -> Pipeline:
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.spec.numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.spec.categorical_features),
            ],
            remainder="drop",
        )
        return Pipeline([("pre", pre), ("model", estimator)])

    # -- data -----------------------------------------------------------------
    def X(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.spec.features].copy()

    def y(self, df: pd.DataFrame):
        return df[self.spec.target].values if self.spec.target else None

    # -- lifecycle ------------------------------------------------------------
    def fit(self, df: pd.DataFrame, **kw):
        self.pipeline.fit(self.X(df), self.y(df))
        return self

    def predict(self, df: pd.DataFrame):
        if self.spec.task in ("classification", "anomaly", "anomaly_classification"):
            return (self.predict_proba(df)[:, 1] >= self.threshold).astype(int)
        return self.pipeline.predict(self.X(df))

    def predict_proba(self, df: pd.DataFrame):
        return self.pipeline.predict_proba(self.X(df))

    def feature_importance(self) -> dict:
        model = self.pipeline.named_steps["model"]
        try:
            names = self.pipeline.named_steps["pre"].get_feature_names_out()
        except Exception:
            names = self.spec.features
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            return {}
        # collapse one-hot back to the base feature it came from
        out: dict[str, float] = {}
        for n, v in zip(names, imp):
            base = n.split("__")[-1]
            for f in self.spec.features:
                if base == f or base.startswith(f + "_"):
                    out[f] = out.get(f, 0.0) + float(v)
                    break
        s = sum(out.values()) or 1.0
        return {k: round(v / s, 4) for k, v in sorted(out.items(), key=lambda t: -t[1])}

    def evaluate(self, df: pd.DataFrame) -> dict:
        y = self.y(df)
        if self.spec.task in ("classification", "anomaly", "anomaly_classification"):
            from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                         recall_score, roc_auc_score)
            pred = self.predict(df)
            proba = self.predict_proba(df)[:, 1]
            return {
                "accuracy": round(float(accuracy_score(y, pred)), 4),
                "f1": round(float(f1_score(y, pred)), 4),
                "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
                "auc": round(float(roc_auc_score(y, proba)), 4),
            }
        if self.spec.task == "regression":
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            pred = self.predict(df)
            # MAPE only over non-trivial targets (near-zero values explode the ratio)
            mask = np.abs(y) > 0.05 * np.mean(np.abs(y))
            mape = float(np.mean(np.abs((y[mask] - pred[mask]) / y[mask]))) if mask.any() else float("nan")
            return {
                "mae": round(float(mean_absolute_error(y, pred)), 3),
                "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 3),
                "r2": round(float(r2_score(y, pred)), 4),
                "mape": round(mape, 4),
            }
        return {}
