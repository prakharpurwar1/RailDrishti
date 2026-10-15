"""The five predictive models of the passenger-facing layer."""
from __future__ import annotations

from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              IsolationForest)

from .base import GovernedModel, ModelSpec

# ---------------------------------------------------------------------------
# 1. Ticket confirmation predictor
# ---------------------------------------------------------------------------
CONFIRMATION_SPEC = ModelSpec(
    name="confirmation_predictor",
    task="classification",
    target="will_confirm",
    numeric_features=["booking_days_before", "month", "season_index", "is_peak_day",
                      "route_popularity", "capacity",
                      "wl_position", "rac_position", "distance_km"],
    categorical_features=["train_type", "route", "travel_class", "quota"],
    protected_attrs=["quota", "travel_class"],
    drift_features=["booking_days_before", "wl_position", "route_popularity", "season_index"],
    sla={"min_auc": 0.75, "min_accuracy": 0.70, "max_latency_ms": 250},
)


def build_confirmation_model() -> GovernedModel:
    est = GradientBoostingClassifier(n_estimators=220, max_depth=3, learning_rate=0.08,
                                     subsample=0.9, random_state=42)
    return GovernedModel(CONFIRMATION_SPEC, est)


# ---------------------------------------------------------------------------
# 2. Delay & disruption predictor
# ---------------------------------------------------------------------------
DELAY_SPEC = ModelSpec(
    name="delay_predictor",
    task="regression",
    target="arrival_delay_min",
    numeric_features=["distance_km", "month", "season_index", "day_of_week",
                      "scheduled_dep_hour", "weather_severity", "section_congestion",
                      "historical_avg_delay"],
    categorical_features=["train_type", "route"],
    protected_attrs=["train_type"],
    drift_features=["section_congestion", "weather_severity", "historical_avg_delay"],
    sla={"max_mae_min": 45, "max_latency_ms": 250},
)


def build_delay_model() -> GovernedModel:
    est = GradientBoostingRegressor(n_estimators=300, learning_rate=0.06, max_depth=4,
                                    subsample=0.9, random_state=42)
    return GovernedModel(DELAY_SPEC, est)


# ---------------------------------------------------------------------------
# 3. Dynamic fare predictor
# ---------------------------------------------------------------------------
FARE_SPEC = ModelSpec(
    name="fare_predictor",
    task="regression",
    target="fare",
    numeric_features=["days_before", "demand_index", "season_index", "base_fare", "distance_km"],
    categorical_features=["route", "travel_class"],
    protected_attrs=["travel_class"],
    drift_features=["demand_index", "days_before", "base_fare"],
    sla={"max_mape": 0.12, "max_latency_ms": 250},
)


def build_fare_model() -> GovernedModel:
    est = GradientBoostingRegressor(n_estimators=250, max_depth=4, learning_rate=0.08,
                                    random_state=42)
    return GovernedModel(FARE_SPEC, est)


# ---------------------------------------------------------------------------
# 4. Fraud / bot detector  (classifier + unsupervised anomaly scorer)
# ---------------------------------------------------------------------------
FRAUD_SPEC = ModelSpec(
    name="fraud_detector",
    task="anomaly_classification",
    target="is_fraud",
    numeric_features=["account_age_days", "bookings_last_24h", "distinct_pnrs", "ip_count",
                      "device_count", "failed_payments", "avg_gap_sec"],
    categorical_features=["payment_method"],
    protected_attrs=[],
    drift_features=["bookings_last_24h", "ip_count", "failed_payments"],
    sla={"min_recall": 0.80, "max_fpr": 0.05, "max_latency_ms": 200},
)


def build_fraud_model() -> GovernedModel:
    est = GradientBoostingClassifier(n_estimators=250, learning_rate=0.08, max_depth=3,
                                     subsample=0.9, random_state=42)
    return GovernedModel(FRAUD_SPEC, est)


def build_fraud_anomaly(normal_df, spec: ModelSpec = FRAUD_SPEC) -> IsolationForest:
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(normal_df[spec.numeric_features].values)
    return iso


# ---------------------------------------------------------------------------
# 5. Train recommender (click/utility ranker)
# ---------------------------------------------------------------------------
RECO_SPEC = ModelSpec(
    name="recommender",
    task="classification",           # scores candidates for ranking
    target="clicked",
    numeric_features=["budget", "max_duration_hr", "cand_fare", "cand_duration_hr",
                      "cand_punctuality", "cand_availability", "class_match",
                      "within_budget", "within_duration"],
    categorical_features=["preferred_class", "priority", "cand_class", "cand_train_type"],
    protected_attrs=["cand_class", "priority"],
    drift_features=["budget", "max_duration_hr", "cand_punctuality"],
    sla={"min_ndcg_at_5": 0.60, "max_latency_ms": 300},
)


def build_recommender() -> GovernedModel:
    est = GradientBoostingClassifier(n_estimators=220, learning_rate=0.08, max_depth=3,
                                     subsample=0.9, random_state=42)
    return GovernedModel(RECO_SPEC, est)


ALL_BUILDERS = {
    "confirmation_predictor": build_confirmation_model,
    "delay_predictor": build_delay_model,
    "fare_predictor": build_fare_model,
    "fraud_detector": build_fraud_model,
    "recommender": build_recommender,
}

DATA_FILES = {
    "confirmation_predictor": "bookings",
    "delay_predictor": "train_runs",
    "fare_predictor": "fares",
    "fraud_detector": "users",
    "recommender": "search_sessions",
}
