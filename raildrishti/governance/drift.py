"""Drift detection: data drift (PSI / KS), concept drift and behaviour change."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI between a reference and a current sample. <0.1 stable, 0.1-0.25 watch, >0.25 drift."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(breakpoints) < 3:
        return 0.0
    e_perc, _ = np.histogram(expected, bins=breakpoints)
    a_perc, _ = np.histogram(actual, bins=breakpoints)
    e_perc = np.clip(e_perc / max(e_perc.sum(), 1), 1e-6, None)
    a_perc = np.clip(a_perc / max(a_perc.sum(), 1), 1e-6, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def ks_drift(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic and p-value."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 5 or len(actual) < 5:
        return 0.0, 1.0
    stat, p = stats.ks_2samp(expected, actual)
    return float(stat), float(p)


def _severity(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "watch"
    return "drift"


def detect_data_drift(reference: pd.DataFrame, current: pd.DataFrame,
                      features: list[str]) -> dict:
    """Per-feature drift report with PSI, KS and a severity flag."""
    report = {"features": {}, "n_drifted": 0, "max_psi": 0.0, "severity": "stable"}
    for f in features:
        if f not in reference.columns or f not in current.columns:
            continue
        psi = population_stability_index(reference[f].values, current[f].values)
        ks, p = ks_drift(reference[f].values, current[f].values)
        sev = _severity(psi)
        report["features"][f] = {
            "psi": round(psi, 4), "ks_stat": round(ks, 4), "ks_pvalue": round(p, 5),
            "severity": sev,
            "ref_mean": round(float(np.nanmean(reference[f].values)), 4),
            "cur_mean": round(float(np.nanmean(current[f].values)), 4),
        }
        if sev == "drift":
            report["n_drifted"] += 1
        report["max_psi"] = max(report["max_psi"], psi)
    report["max_psi"] = round(report["max_psi"], 4)
    if report["n_drifted"] >= 2:
        report["severity"] = "drift"
    elif report["n_drifted"] == 1 or report["max_psi"] >= 0.10:
        report["severity"] = "watch"
    return report


def detect_concept_drift(perf_history: list[float], baseline: float,
                         tolerance: float = 0.05) -> dict:
    """Concept drift = the input->output relationship changed; shows up as a
    sustained drop in a rolling performance metric below its baseline."""
    perf_history = [p for p in perf_history if p is not None]
    if len(perf_history) < 3:
        return {"concept_drift": False, "recent_mean": None, "drop": 0.0}
    recent = float(np.mean(perf_history[-3:]))
    drop = float(baseline - recent)
    return {
        "concept_drift": drop > tolerance,
        "baseline": round(baseline, 4),
        "recent_mean": round(recent, 4),
        "drop": round(drop, 4),
    }


def detect_behavior_change(ref_actions: pd.Series, cur_actions: pd.Series) -> dict:
    """User-behaviour change: shift in the distribution of actions/intents."""
    ref = ref_actions.value_counts(normalize=True)
    cur = cur_actions.value_counts(normalize=True)
    idx = ref.index.union(cur.index)
    ref = ref.reindex(idx, fill_value=0.0)
    cur = cur.reindex(idx, fill_value=0.0)
    tvd = float(0.5 * np.abs(ref - cur).sum())  # total variation distance
    return {"tvd": round(tvd, 4), "changed": tvd > 0.15,
            "ref_dist": ref.round(3).to_dict(), "cur_dist": cur.round(3).to_dict()}
