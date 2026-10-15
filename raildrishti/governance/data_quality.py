"""Data quality validation gate.

Runs before inference: catches missing values, out-of-range values, type
violations, duplicate keys and statistical outliers so poor-quality data never
reaches a model ("prevent poor-quality predictions").
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def validate(df: pd.DataFrame, schema: dict | None = None,
             key: str | None = None, z_threshold: float = 6.0) -> dict:
    """schema = {col: {"min":.., "max":.., "dtype":"numeric|category|string"}}"""
    schema = schema or {}
    n = len(df)
    report = {"n_rows": n, "checks": {}, "issues": [], "quality_score": 1.0}

    # 1. missing values
    missing = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    miss_rate = sum(missing.values()) / max(n * max(len(df.columns), 1), 1)
    report["checks"]["missing"] = {"per_column": missing, "rate": round(miss_rate, 4)}
    if missing:
        report["issues"].append(f"missing values in {list(missing)}")

    # 2. range / dtype violations
    violations = {}
    for col, rule in schema.items():
        if col not in df.columns:
            violations[col] = "absent"
            continue
        s = df[col]
        if "min" in rule:
            bad = int((s < rule["min"]).sum())
            if bad:
                violations[col] = f"{bad} below min {rule['min']}"
        if "max" in rule:
            bad = int((s > rule["max"]).sum())
            if bad:
                violations[col] = f"{bad} above max {rule['max']}"
        if rule.get("dtype") == "numeric" and not pd.api.types.is_numeric_dtype(s):
            violations[col] = "non-numeric"
    report["checks"]["range"] = violations
    if violations:
        report["issues"].append(f"range/type violations: {list(violations)}")

    # 3. duplicate keys
    if key and key in df.columns:
        dup = int(df[key].duplicated().sum())
        report["checks"]["duplicates"] = {"key": key, "count": dup}
        if dup:
            report["issues"].append(f"{dup} duplicate {key}")

    # 4. outliers (z-score on numeric cols)
    outliers = {}
    for c in _numeric_cols(df):
        s = df[c].astype(float)
        if s.std() == 0 or s.isna().all():
            continue
        z = np.abs((s - s.mean()) / s.std())
        cnt = int((z > z_threshold).sum())
        if cnt:
            outliers[c] = cnt
    report["checks"]["outliers"] = outliers
    if outliers:
        report["issues"].append(f"outliers in {list(outliers)}")

    # composite score
    score = 1.0
    score -= min(miss_rate * 2.0, 0.5)
    score -= min(len(violations) * 0.1, 0.3)
    score -= min(sum(outliers.values()) / max(n, 1), 0.2)
    report["quality_score"] = round(max(score, 0.0), 4)
    report["passed"] = report["quality_score"] >= 0.75 and not violations
    return report


def flag_bad_rows(df: pd.DataFrame, schema: dict) -> pd.Series:
    """Boolean mask of rows that violate the schema (used to quarantine input)."""
    bad = pd.Series(False, index=df.index)
    for col, rule in schema.items():
        if col not in df.columns:
            continue
        if "min" in rule:
            bad |= df[col] < rule["min"]
        if "max" in rule:
            bad |= df[col] > rule["max"]
    return bad
