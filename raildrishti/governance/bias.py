"""Fairness / bias evaluation and hallucination monitoring."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rates(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict:
    out = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 5:
            continue
        yt, yp = y_true[m], y_pred[m]
        tp = np.sum((yp == 1) & (yt == 1))
        fn = np.sum((yp == 0) & (yt == 1))
        fp = np.sum((yp == 1) & (yt == 0))
        tn = np.sum((yp == 0) & (yt == 0))
        sel = float(np.mean(yp))
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        out[str(g)] = {"n": int(m.sum()), "selection_rate": round(sel, 4),
                       "tpr": round(float(tpr), 4), "fpr": round(float(fpr), 4)}
    return out


def evaluate_bias(y_true, y_pred, protected: pd.Series, favorable_label: int = 1) -> dict:
    """Group fairness metrics for one protected attribute.

    - demographic parity difference  (max-min selection rate)
    - equal opportunity difference   (max-min true-positive rate)
    - disparate impact ratio         (min/max selection rate)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(protected)
    rates = _rates(y_true, y_pred, groups)
    if len(rates) < 2:
        return {"attribute": protected.name, "groups": rates, "flagged": False}
    sels = [v["selection_rate"] for v in rates.values()]
    tprs = [v["tpr"] for v in rates.values()]
    dp_diff = float(max(sels) - min(sels))
    eo_diff = float(max(tprs) - min(tprs))
    di_ratio = float(min(sels) / max(sels)) if max(sels) > 0 else 1.0
    flagged = (dp_diff > 0.10) or (di_ratio < 0.80) or (eo_diff > 0.10)
    return {
        "attribute": protected.name,
        "groups": rates,
        "demographic_parity_diff": round(dp_diff, 4),
        "equal_opportunity_diff": round(eo_diff, 4),
        "disparate_impact_ratio": round(di_ratio, 4),
        "flagged": bool(flagged),
        "severity": "high" if (dp_diff > 0.20 or di_ratio < 0.60) else ("medium" if flagged else "low"),
    }


# ---------------------------------------------------------------------------
# Hallucination / groundedness monitoring for generative (support) answers
# ---------------------------------------------------------------------------
def groundedness_score(answer: str, evidence: list[str]) -> float:
    """Fraction of the answer's content tokens supported by the retrieved evidence.
    A lightweight, dependency-free proxy for groundedness / hallucination risk."""
    import re
    toks = set(re.findall(r"[a-zA-Z0-9]+", answer.lower()))
    stop = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on",
            "for", "you", "your", "it", "this", "that", "will", "can", "be", "at"}
    toks -= stop
    if not toks:
        return 1.0
    ev = " ".join(evidence).lower()
    supported = sum(1 for t in toks if t in ev)
    return round(supported / len(toks), 4)


def hallucination_report(answer: str, evidence: list[str], threshold: float = 0.75) -> dict:
    g = groundedness_score(answer, evidence)
    return {"groundedness": g, "hallucination": g < threshold,
            "risk": "high" if g < 0.5 else ("medium" if g < threshold else "low")}
