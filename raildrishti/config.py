"""Central configuration for RailDrishti.

RailDrishti = AI-Powered Intelligent Railway Travel & Autonomous AI Governance Platform.
This module holds paths, constants and the model registry manifest used across the codebase.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(os.environ.get("RAILDRISHTI_ARTIFACTS", ROOT / "artifacts"))
DATA_DIR = ARTIFACTS / "data"
MODEL_DIR = ARTIFACTS / "models"
GOV_DIR = ARTIFACTS / "governance"
AUDIT_LOG = GOV_DIR / "audit_trail.jsonl"

for _p in (ARTIFACTS, DATA_DIR, MODEL_DIR, GOV_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Domain constants (Indian Railways flavoured)
# ---------------------------------------------------------------------------
TRAVEL_CLASSES = ["1A", "2A", "3A", "SL", "CC", "2S"]
# Relative demand/popularity weight per class (SL & 3A are the workhorses)
CLASS_DEMAND = {"1A": 0.4, "2A": 0.9, "3A": 1.6, "SL": 2.0, "CC": 1.2, "2S": 1.4}
CLASS_BASE_FARE = {"1A": 4200, "2A": 2600, "3A": 1800, "SL": 750, "CC": 1100, "2S": 420}

QUOTAS = ["GN", "TQ", "LD", "PT", "SS"]  # General, Tatkal, Ladies, Pooled, Senior

TRAIN_TYPES = ["Rajdhani", "Shatabdi", "Duronto", "Superfast", "Express", "Passenger", "MEMU"]
# Punctuality prior by train type (Rajdhani most punctual, Passenger least)
TRAIN_TYPE_PUNCTUALITY = {
    "Rajdhani": 0.90, "Shatabdi": 0.92, "Duronto": 0.85,
    "Superfast": 0.80, "Express": 0.72, "Passenger": 0.55, "MEMU": 0.60,
}

# Popular inter-city corridors (route, distance_km, base popularity 0-1)
ROUTES = [
    ("NDLS-BCT", "New Delhi - Mumbai CSMT", 1384, 0.98),
    ("NDLS-HWH", "New Delhi - Howrah", 1451, 0.95),
    ("CSMT-SBC", "Mumbai CSMT - Bengaluru", 1213, 0.90),
    ("NDLS-MAS", "New Delhi - Chennai", 2175, 0.88),
    ("PUNE-NDLS", "Pune - New Delhi", 1515, 0.86),
    ("SBC-MAS", "Bengaluru - Chennai", 362, 0.93),
    ("HWH-MAS", "Howrah - Chennai", 1662, 0.80),
    ("NDLS-LKO", "New Delhi - Lucknow", 512, 0.84),
    ("BCT-ADI", "Mumbai - Ahmedabad", 493, 0.82),
    ("NDLS-ASR", "New Delhi - Amritsar", 448, 0.78),
    ("SC-BZA", "Secunderabad - Vijayawada", 361, 0.70),
    ("MAS-KCVL", "Chennai - Kochuveli", 926, 0.66),
]

STATIONS = ["NDLS", "BCT", "HWH", "MAS", "SBC", "PUNE", "ADI", "LKO", "ASR", "BZA", "SC", "KCVL"]

MONSOON_MONTHS = {6, 7, 8, 9}
FESTIVAL_MONTHS = {10, 11, 3}  # Diwali/Puja, Chhath, Holi window

# ---------------------------------------------------------------------------
# Model registry manifest: the single source of truth for what the
# governance layer must monitor. Every entry becomes a governed model.
# ---------------------------------------------------------------------------
MODEL_MANIFEST = {
    "confirmation_predictor": {
        "module": "ticket_confirmation",
        "task": "classification",
        "target": "final_status",
        "sla": {"min_auc": 0.75, "min_accuracy": 0.70, "max_latency_ms": 250},
        "protected_attrs": ["quota", "travel_class"],
        "drift_features": ["booking_days_before", "wl_position", "route_popularity", "season_index"],
    },
    "delay_predictor": {
        "module": "delay_disruption",
        "task": "regression",
        "target": "arrival_delay_min",
        "sla": {"max_mae_min": 45, "max_latency_ms": 250},
        "protected_attrs": ["train_type"],
        "drift_features": ["section_congestion", "weather_severity", "historical_avg_delay"],
    },
    "fare_predictor": {
        "module": "dynamic_fare",
        "task": "regression",
        "target": "fare",
        "sla": {"max_mape": 0.12, "max_latency_ms": 250},
        "protected_attrs": ["travel_class"],
        "drift_features": ["demand_index", "days_before", "base_fare"],
    },
    "recommender": {
        "module": "recommendation",
        "task": "ranking",
        "target": "clicked",
        "sla": {"min_ndcg_at_5": 0.60, "max_latency_ms": 300},
        "protected_attrs": ["cand_class", "priority"],
        "drift_features": ["budget", "preferred_duration", "punctuality_prior"],
    },
    "fraud_detector": {
        "module": "fraud_security",
        "task": "anomaly_classification",
        "target": "is_fraud",
        "sla": {"min_recall": 0.80, "max_fpr": 0.05, "max_latency_ms": 200},
        "protected_attrs": [],
        "drift_features": ["bookings_last_24h", "ip_count", "failed_payments"],
    },
    "support_assistant": {
        "module": "customer_support",
        "task": "generation",
        "target": None,
        "sla": {"max_hallucination_rate": 0.03, "min_groundedness": 0.90, "max_latency_ms": 1500},
        "protected_attrs": ["language"],
        "drift_features": ["query_intent_distribution"],
    },
    "travel_planner_agent": {
        "module": "travel_planner",
        "task": "agentic",
        "target": None,
        "sla": {"min_task_success": 0.85, "max_latency_ms": 3000},
        "protected_attrs": [],
        "drift_features": ["tool_call_distribution"],
    },
}

SUPPORT_LANGUAGES = ["en", "hi", "ta", "te", "bn", "mr", "kn", "gu"]
