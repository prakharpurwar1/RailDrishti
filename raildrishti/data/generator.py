"""Synthetic Indian Railways data generator.

Produces realistic, signal-bearing datasets so every ML model in RailDrishti can be
trained and evaluated without depending on private railway data:

  bookings.csv          -> ticket confirmation predictor
  train_runs.csv        -> delay & disruption predictor
  station_congestion.csv-> station congestion (feeds delay model + alerts)
  fares.csv             -> dynamic fare predictor
  users.csv             -> fraud / bot detector
  search_sessions.csv   -> train recommendation engine

The latent processes encode genuine domain structure (waiting-list conversion,
monsoon delays, dynamic pricing, bot behaviour) so models learn real relationships.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    CLASS_DEMAND, CLASS_BASE_FARE, QUOTAS, TRAIN_TYPES,
    TRAIN_TYPE_PUNCTUALITY, ROUTES, STATIONS, MONSOON_MONTHS,
    FESTIVAL_MONTHS, RANDOM_SEED, DATA_DIR,
)

CLASS_CAPACITY = {"1A": 100, "2A": 200, "3A": 350, "SL": 420, "CC": 250, "2S": 500}
BERTHS = ["LB", "MB", "UB", "SL", "SU", "WS"]


def _rng(seed=None):
    return np.random.default_rng(RANDOM_SEED if seed is None else seed)


def _season_index(month: np.ndarray) -> np.ndarray:
    """0 = normal, ~1 = festival peak, ~0.6 = monsoon disruption."""
    s = np.zeros_like(month, dtype=float)
    s += np.isin(month, list(FESTIVAL_MONTHS)) * 0.9
    s += np.isin(month, list(MONSOON_MONTHS)) * 0.5
    return np.clip(s, 0, 1.5)


# ---------------------------------------------------------------------------
# 1. Bookings  (ticket confirmation predictor)
# ---------------------------------------------------------------------------
def generate_bookings(n: int = 60_000, seed=None) -> pd.DataFrame:
    rng = _rng(seed)
    route_idx = rng.integers(0, len(ROUTES), n)
    routes = [ROUTES[i] for i in route_idx]
    route_code = np.array([r[0] for r in routes])
    route_pop = np.array([r[3] for r in routes])
    distance_km = np.array([r[2] for r in routes])

    cls = rng.choice(list(CLASS_DEMAND), n, p=np.array(list(CLASS_DEMAND.values())) / sum(CLASS_DEMAND.values()))
    quota = rng.choice(QUOTAS, n, p=[0.72, 0.15, 0.06, 0.04, 0.03])
    train_type = rng.choice(TRAIN_TYPES, n, p=[0.06, 0.06, 0.08, 0.18, 0.34, 0.18, 0.10])

    booking_days_before = np.clip(rng.exponential(22, n), 0, 120).astype(int)
    month = rng.integers(1, 13, n)
    season = _season_index(month)
    is_peak_day = (rng.random(n) < 0.25 + 0.2 * (season > 0.5)).astype(int)

    capacity = np.array([CLASS_CAPACITY[c] for c in cls], dtype=float)
    class_demand = np.array([CLASS_DEMAND[c] for c in cls], dtype=float)

    # Load pressure: demand relative to supply, rising as the journey approaches.
    proximity = 1.0 - booking_days_before / 160.0            # 0.25 (early) .. 1.0 (last minute)
    tatkal = (quota == "TQ").astype(float) * 0.35
    load = (route_pop * class_demand * (1 + 0.6 * season) * (1 + 0.25 * is_peak_day) * (0.4 + 0.6 * proximity))
    occ = load * rng.lognormal(0, 0.22, n) + tatkal
    occupancy_ratio = np.clip(occ, 0.05, 3.0)

    wl_position = np.zeros(n, dtype=int)
    rac_position = np.zeros(n, dtype=int)
    booked_confirmed = occupancy_ratio <= 0.98
    rac_band = (occupancy_ratio > 0.98) & (occupancy_ratio <= 1.06)
    wl_band = occupancy_ratio > 1.06
    wl_position[wl_band] = np.round((occupancy_ratio[wl_band] - 1.0) * capacity[wl_band] * 0.9).astype(int)
    rac_position[rac_band] = np.round((occupancy_ratio[rac_band] - 0.98) * capacity[rac_band]).astype(int)
    wl_position = np.clip(wl_position, 0, 400)

    # Conversion: cancellations free up berths; early bookings see more churn.
    cancel_rate = 0.10 + 0.22 * (booking_days_before / 120.0) + 0.08 * (season > 0.5)
    free_seats = capacity * cancel_rate * (0.6 + 0.4 * route_pop) * rng.lognormal(0, 0.3, n)

    final_status = np.empty(n, dtype=object)
    final_status[booked_confirmed] = "CONFIRMED"
    # RAC usually clears
    rac_confirm = rac_band & (rng.random(n) < 0.9)
    final_status[rac_confirm] = "CONFIRMED"
    final_status[rac_band & ~rac_confirm] = "RAC"
    wl_confirm = wl_band & (wl_position <= free_seats)
    final_status[wl_confirm] = "CONFIRMED"
    final_status[wl_band & ~wl_confirm] = "WAITLISTED"
    # A small share of confirmed bookings cancel outright
    cancelled = (final_status == "CONFIRMED") & (rng.random(n) < 0.05)
    final_status[cancelled] = "CANCELLED"

    berth = np.where(final_status == "CONFIRMED", rng.choice(BERTHS, n), "")

    df = pd.DataFrame({
        "pnr": [f"{rng.integers(10**9, 10**10)}" for _ in range(n)],
        "train_no": rng.integers(12001, 22999, n),
        "train_type": train_type,
        "route": route_code,
        "distance_km": distance_km,
        "travel_class": cls,
        "quota": quota,
        "booking_days_before": booking_days_before,
        "month": month,
        "season_index": np.round(season, 3),
        "is_peak_day": is_peak_day,
        "route_popularity": route_pop,
        "capacity": capacity.astype(int),
        "occupancy_ratio": np.round(occupancy_ratio, 4),
        "wl_position": wl_position,
        "rac_position": rac_position,
        "final_status": final_status,
        "will_confirm": (final_status == "CONFIRMED").astype(int),
        "berth_status": berth,
    })
    return df


# ---------------------------------------------------------------------------
# 2. Train runs (delay predictor) + station congestion
# ---------------------------------------------------------------------------
def generate_train_runs(n: int = 12_000, seed=None) -> pd.DataFrame:
    rng = _rng(seed)
    route_idx = rng.integers(0, len(ROUTES), n)
    routes = [ROUTES[i] for i in route_idx]
    route_code = np.array([r[0] for r in routes])
    distance_km = np.array([r[2] for r in routes], dtype=float)
    route_pop = np.array([r[3] for r in routes])

    train_type = rng.choice(TRAIN_TYPES, n, p=[0.06, 0.06, 0.08, 0.18, 0.34, 0.18, 0.10])
    punctuality = np.array([TRAIN_TYPE_PUNCTUALITY[t] for t in train_type])
    month = rng.integers(1, 13, n)
    season = _season_index(month)
    day_of_week = rng.integers(0, 7, n)
    sched_dep_hour = rng.integers(0, 24, n)

    weather_severity = np.clip(rng.beta(2, 5, n) + 0.35 * (season > 0.4), 0, 1)
    section_congestion = np.clip(route_pop * rng.beta(3, 3, n) + 0.15 * (sched_dep_hour // 6 == 3), 0, 1)
    historical_avg_delay = (1 - punctuality) * 70 + distance_km / 40 + rng.normal(0, 5, n)
    historical_avg_delay = np.clip(historical_avg_delay, 0, None)

    base = (1 - punctuality) * 60
    delay = (
        base
        + 0.006 * distance_km
        + 20 * section_congestion
        + 22 * weather_severity
        + 6 * (sched_dep_hour >= 18)
        + rng.normal(0, 7, n)
    )
    delay = np.clip(delay, 0, None)

    df = pd.DataFrame({
        "run_id": np.arange(n),
        "train_no": rng.integers(12001, 22999, n),
        "train_type": train_type,
        "route": route_code,
        "distance_km": distance_km,
        "month": month,
        "season_index": np.round(season, 3),
        "day_of_week": day_of_week,
        "scheduled_dep_hour": sched_dep_hour,
        "weather_severity": np.round(weather_severity, 3),
        "section_congestion": np.round(section_congestion, 3),
        "historical_avg_delay": np.round(historical_avg_delay, 2),
        "arrival_delay_min": np.round(delay, 1),
        "departure_delay_min": np.round(delay * rng.uniform(0.85, 1.05, n), 1),
        "is_delayed_30plus": (delay >= 30).astype(int),
    })
    return df


def generate_station_congestion(seed=None) -> pd.DataFrame:
    rng = _rng(seed)
    rows = []
    for st in STATIONS:
        base = rng.uniform(0.2, 0.7)
        for hour in range(24):
            peak = 0.35 if hour in (8, 9, 17, 18, 19, 20) else 0.0
            cong = np.clip(base + peak + rng.normal(0, 0.06), 0, 1)
            rows.append({
                "station": st, "hour": hour,
                "footfall_index": round(float(cong * rng.uniform(0.8, 1.2) + 0.05), 3),
                "congestion_level": round(float(cong), 3),
                "platform_utilization": round(float(np.clip(cong * rng.uniform(0.7, 1.1), 0, 1)), 3),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Fares (dynamic fare predictor)
# ---------------------------------------------------------------------------
def generate_fares(n: int = 40_000, seed=None) -> pd.DataFrame:
    rng = _rng(seed)
    route_idx = rng.integers(0, len(ROUTES), n)
    routes = [ROUTES[i] for i in route_idx]
    route_code = np.array([r[0] for r in routes])
    distance_km = np.array([r[2] for r in routes], dtype=float)
    route_pop = np.array([r[3] for r in routes])
    cls = rng.choice(list(CLASS_BASE_FARE), n)
    days_before = rng.integers(0, 61, n)
    demand_index = np.clip(route_pop * rng.beta(3, 2, n), 0, 1)
    month = rng.integers(1, 13, n)
    season = _season_index(month)

    base_fare = np.array([CLASS_BASE_FARE[c] for c in cls]) * (0.6 + distance_km / 1500)
    surge = 1 + 0.55 * demand_index + 0.45 * (1 - days_before / 60) + 0.2 * (season > 0.5)
    fare = base_fare * surge * rng.lognormal(0, 0.06, n)

    df = pd.DataFrame({
        "route": route_code, "distance_km": distance_km, "travel_class": cls,
        "days_before": days_before, "demand_index": np.round(demand_index, 3),
        "season_index": np.round(season, 3), "base_fare": np.round(base_fare, 1),
        "fare": np.round(fare, 1),
    })
    return df


# ---------------------------------------------------------------------------
# 4. Users (fraud / bot detector)
# ---------------------------------------------------------------------------
def generate_users(n: int = 15_000, seed=None) -> pd.DataFrame:
    rng = _rng(seed)
    account_age_days = np.clip(rng.gamma(2, 120, n), 0, 4000).astype(int)
    bookings_last_24h = np.clip(rng.poisson(1.2, n), 0, 60)
    distinct_pnrs = np.clip(bookings_last_24h + rng.integers(0, 4, n), 0, 80)
    ip_count = np.clip(rng.poisson(1.5, n), 1, 50)
    device_count = np.clip(rng.poisson(1.3, n), 1, 30)
    failed_payments = np.clip(rng.poisson(0.4, n), 0, 25)
    avg_gap_sec = np.clip(rng.lognormal(6.5, 1.2, n), 1, 100000)
    payment_method = rng.choice(["UPI", "CARD", "NETBANKING", "WALLET"], n, p=[0.5, 0.2, 0.2, 0.1])

    # Latent risk: new accounts, burst booking, IP/device fan-out, payment failures, robotic gaps.
    risk = (
        -0.0020 * account_age_days
        + 0.55 * bookings_last_24h
        + 0.42 * ip_count
        + 0.48 * device_count
        + 0.65 * failed_payments
        + 2.6 * (avg_gap_sec < 20)
        + rng.normal(0, 0.45, n)
    )
    thresh = np.quantile(risk, 0.965)
    is_fraud = (risk >= thresh).astype(int)

    df = pd.DataFrame({
        "user_id": np.arange(n),
        "account_age_days": account_age_days,
        "bookings_last_24h": bookings_last_24h,
        "distinct_pnrs": distinct_pnrs,
        "ip_count": ip_count,
        "device_count": device_count,
        "failed_payments": failed_payments,
        "avg_gap_sec": np.round(avg_gap_sec, 1),
        "payment_method": payment_method,
        "risk_score": np.round(1 / (1 + np.exp(-risk)), 4),
        "is_fraud": is_fraud,
    })
    return df


# ---------------------------------------------------------------------------
# 5. Search sessions (recommendation engine)
# ---------------------------------------------------------------------------
def generate_search_sessions(n: int = 40_000, seed=None) -> pd.DataFrame:
    """Each row is a (user request, candidate train) pair with a click label."""
    rng = _rng(seed)
    budget = rng.choice([900, 1500, 2500, 4000, 6000], n)
    preferred_class = rng.choice(["SL", "3A", "2A", "CC"], n)
    max_duration_hr = rng.choice([6, 10, 16, 24, 36], n)
    priority = rng.choice(["price", "punctuality", "comfort", "speed"], n, p=[0.4, 0.25, 0.2, 0.15])

    cand_class = rng.choice(["SL", "3A", "2A", "CC", "1A"], n)
    cand_fare = np.clip(rng.lognormal(7.4, 0.5, n), 300, 8000)
    cand_duration = np.clip(rng.lognormal(2.3, 0.5, n), 2, 40)
    cand_punctuality = np.clip(rng.beta(5, 3, n), 0, 1)
    cand_availability = np.clip(rng.beta(3, 2, n), 0, 1)
    cand_train_type = rng.choice(TRAIN_TYPES, n, p=[0.06, 0.06, 0.08, 0.18, 0.34, 0.18, 0.10])

    class_match = (cand_class == preferred_class).astype(float)
    within_budget = (cand_fare <= budget).astype(float)
    within_duration = (cand_duration <= max_duration_hr).astype(float)

    price_score = np.clip(1 - cand_fare / (budget + 1), 0, 1)
    comfort_score = np.array([{"1A": 1.0, "2A": 0.85, "3A": 0.65, "CC": 0.6, "SL": 0.45}[c] for c in cand_class])
    speed_score = np.clip(1 - cand_duration / (max_duration_hr + 1), 0, 1)

    W = np.array([
        [0.50, 0.15, 0.20, 0.15],  # price
        [0.15, 0.55, 0.15, 0.15],  # punctuality
        [0.15, 0.15, 0.55, 0.15],  # comfort
        [0.15, 0.20, 0.15, 0.50],  # speed
    ])
    pri_idx = np.array([{"price": 0, "punctuality": 1, "comfort": 2, "speed": 3}[p] for p in priority])
    w = W[pri_idx]
    utility = (
        w[:, 0] * price_score + w[:, 1] * cand_punctuality + w[:, 2] * comfort_score + w[:, 3] * speed_score
        + 0.4 * class_match + 0.5 * within_budget + 0.3 * within_duration + 0.2 * cand_availability
        + rng.normal(0, 0.15, n)
    )
    p = 1 / (1 + np.exp(-(utility - 1.0) * 3))
    clicked = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "budget": budget, "preferred_class": preferred_class, "max_duration_hr": max_duration_hr,
        "priority": priority, "cand_class": cand_class, "cand_fare": np.round(cand_fare, 1),
        "cand_duration_hr": np.round(cand_duration, 2), "cand_punctuality": np.round(cand_punctuality, 3),
        "cand_availability": np.round(cand_availability, 3), "cand_train_type": cand_train_type,
        "class_match": class_match, "within_budget": within_budget, "within_duration": within_duration,
        "clicked": clicked,
    })
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_all(out_dir=DATA_DIR, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "bookings": generate_bookings(),
        "train_runs": generate_train_runs(),
        "station_congestion": generate_station_congestion(),
        "fares": generate_fares(),
        "users": generate_users(),
        "search_sessions": generate_search_sessions(),
    }
    paths = {}
    for name, df in tables.items():
        p = out_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = str(p)
        if verbose:
            print(f"  {name:20s} rows={len(df):>7,}  cols={df.shape[1]}")
    return paths


if __name__ == "__main__":
    print("Generating RailDrishti synthetic datasets ->", DATA_DIR)
    generate_all()
    print("Done.")
