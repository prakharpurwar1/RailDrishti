"""AI Travel Planner Agent.

An agentic assistant that understands a natural-language travel request, then
autonomously calls tools (search -> price -> delay -> confirm -> rank -> plan),
filters candidates against the passenger's hard constraints (budget, arrival
deadline), presents the best itinerary, and books only after explicit approval.

Every step is recorded so the governance layer can validate the agent's decision,
monitor performance and detect abnormal behaviour.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from ..config import CLASS_BASE_FARE, ROUTES, TRAIN_TYPE_PUNCTUALITY, TRAIN_TYPES, RANDOM_SEED

CITY_ALIASES = {
    "NDLS": ["new delhi", "delhi", "ndls"], "BCT": ["mumbai", "bombay", "bct", "csmt"],
    "HWH": ["howrah", "kolkata", "calcutta", "hwh"], "MAS": ["chennai", "madras", "mas"],
    "SBC": ["bengaluru", "bangalore", "sbc"], "PUNE": ["pune", "poona"],
    "ADI": ["ahmedabad", "adi"], "LKO": ["lucknow", "lko"], "ASR": ["amritsar", "asr"],
    "BZA": ["vijayawada", "bza"], "SC": ["secunderabad", "sc"], "KCVL": ["kochuveli", "trivandrum", "kcvl"],
}
CODE_TO_CITY = {
    "NDLS": "Delhi", "BCT": "Mumbai", "HWH": "Howrah", "MAS": "Chennai",
    "SBC": "Bengaluru", "PUNE": "Pune", "ADI": "Ahmedabad", "LKO": "Lucknow",
    "ASR": "Amritsar", "BZA": "Vijayawada", "SC": "Secunderabad", "KCVL": "Kochuveli",
}

COMFORT_RANK = {"1A": 6, "2A": 5, "3A": 4, "CC": 3, "SL": 2, "2S": 1}


def rank_candidates(candidates: list[dict], priority: str | None = None,
                    preferred_class: str | None = None,
                    session_profile: dict | None = None) -> list[dict]:
    """Rank candidates according to preferred class match, availability tier, user priority, and utility score."""
    ranked = [dict(c) for c in candidates]

    # Session-level profile boost
    if session_profile:
        p_class = session_profile.get("preferred_class")
        p_prio = (session_profile.get("priority") or "").lower()
        for c in ranked:
            boost = 0.0
            if p_class and c.get("class") == p_class:
                boost += 0.03
            if p_prio and (c.get("priority") or "").lower() == p_prio:
                boost += 0.02
            if boost > 0:
                c["relevance_score"] = round(min(0.99, c.get("relevance_score", 0.5) + boost), 4)
                c["profile_boost"] = True

    pref_cls = preferred_class or (session_profile.get("preferred_class") if session_profile else None)
    prio = (priority or "").lower()

    def sort_key(c: dict):
        # 1. Preferred class match (options in preferred class must outrank alternate classes)
        is_pref = 1 if (pref_cls and c.get("class") == pref_cls) else 0

        # 2. Availability tier (confirmed/good availability >= 30% outranks high waitlist < 30%)
        avail = c.get("availability", 0.0)
        good_avail = 1 if avail >= 0.30 else 0

        # 3. Selected user priority
        if prio == "price":
            metric = c.get("fare", 999999)
            return (-is_pref, -good_avail, metric, -c.get("relevance_score", 0))
        elif prio == "speed":
            metric = c.get("duration_hr", 999.0)
            return (-is_pref, -good_avail, metric, -c.get("relevance_score", 0))
        elif prio == "punctuality":
            return (-is_pref, -good_avail, -c.get("punctuality", 0.0), c.get("delay_min", 999.0), -c.get("relevance_score", 0))
        elif prio == "comfort":
            metric = -COMFORT_RANK.get(c.get("class"), 0)
            return (-is_pref, -good_avail, metric, -c.get("relevance_score", 0))
        else:
            return (-is_pref, -good_avail, -c.get("relevance_score", 0))

    ranked.sort(key=sort_key)
    return ranked


class PlannerAgent:
    def __init__(self, platform):
        self.platform = platform
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.last_steps: list[dict] = []

    # -- tool log ------------------------------------------------------------
    def _log(self, tool: str, args: dict, result_summary: str):
        self.last_steps.append({"tool": tool, "args": args, "result": result_summary,
                                "ts": time.time()})

    # -- NL understanding ----------------------------------------------------
    def parse(self, query: str) -> dict:
        q = query.lower()
        cities = []
        for code, aliases in CITY_ALIASES.items():
            if any(a in q for a in aliases):
                cities.append(code)
        # order of appearance in the text
        cities = sorted(set(cities), key=lambda c: min(q.find(a) for a in CITY_ALIASES[c] if a in q))
        src = cities[0] if cities else None
        dst = cities[1] if len(cities) > 1 else None

        budget = None
        m = re.search(r"(?:under|below|within|upto|up to|max)\s*(?:rs\.?|₹|inr)?\s*(\d{3,6})", q)
        if not m:
            m = re.search(r"(?:rs\.?|₹|inr)\s*(\d{3,6})", q)
        if m:
            budget = int(m.group(1))

        deadline = None
        m = re.search(r"(?:reach|arrive)\s*(?:before|by|around)?\s*(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?", q)
        if m:
            h = int(m.group(1))
            ampm = m.group(3)
            if ampm == "pm" and h < 12:
                h += 12
            elif ampm == "am" and h == 12:
                h = 0
            deadline = h

        when = "tomorrow" if "tomorrow" in q else ("today" if "today" in q else ("next week" if "next week" in q else "unspecified"))
        cls = None
        for c in ["1a", "2a", "3a", "sl", "cc", "2s"]:
            if re.search(rf"\b{c}\b", q):
                cls = c.upper()

        explicit_priority = None
        if re.search(r"\b(cheapest|cheap|lowest fare|affordable|low fare|economical)\b", q):
            explicit_priority = "price"
        elif re.search(r"\b(fastest|quickest|shortest)\b", q):
            explicit_priority = "speed"
        elif re.search(r"\b(punctual|on time|on-time|reliable)\b", q):
            explicit_priority = "punctuality"

        priority = explicit_priority or ("price" if budget else ("punctuality" if deadline else "comfort"))

        constraints = {
            "from": src, "to": dst, "budget": budget, "arrive_by_hour": deadline,
            "when": when, "preferred_class": cls, "priority": priority,
            "explicit_priority": explicit_priority,
        }
        self._log("parse_request", {"query": query}, str(constraints))
        return constraints

    # -- tools ---------------------------------------------------------------
    def _route(self, src: str, dst: str):
        def norm(c):
            return "BCT" if c in ("BCT", "CSMT") else c
        s_norm, d_norm = norm(src), norm(dst)
        for code, name, dist, pop in ROUTES:
            a, b = code.split("-")
            if {norm(a), norm(b)} == {s_norm, d_norm}:
                return {"code": code, "name": name, "distance_km": dist, "popularity": pop,
                        "reversed": (norm(a) != s_norm)}
        return None

    def generate_candidates(self, constraints: dict, n: int = 40) -> list[dict]:
        route = self._route(constraints["from"], constraints["to"])
        if route is None:
            return []
        fare_model = self.platform.get_model("fare_predictor")
        delay_model = self.platform.get_model("delay_predictor")
        rec_model = self.platform.get_model("recommender")
        pcls = constraints.get("preferred_class") or "3A"

        all_classes = ["2S", "SL", "CC", "3A", "2A", "1A"]
        # Guarantee coverage across all travel classes and whole-day departure hours (including evening & night)
        base_dep_hours = [5, 9, 13, 17, 20, 22]
        slots = []
        for c in all_classes:
            for h in base_dep_hours:
                slots.append((c, h))
        # Deep whole-day coverage for the preferred class (at least 12-14 options across morning, afternoon, evening, night)
        pref_hours = [6, 8, 10, 12, 14, 16, 18, 19, 21, 23]
        for h in pref_hours:
            slots.append((pcls, h))
        # Additional coverage for other classes
        extra_slots = [
            ("2S", 18), ("2S", 21),
            ("SL", 19), ("SL", 21), ("SL", 23),
            ("CC", 7), ("CC", 18),
            ("2A", 18), ("2A", 20),
            ("1A", 18), ("1A", 20)
        ]
        for s in extra_slots:
            if s[0] != pcls:
                slots.append(s)

        passengers = int(constraints.get("passengers") or 1)

        candidates = []
        for cls, dep_hour in slots:
            if cls in ("1A", "2A", "3A"):
                ttype = self.rng.choice(["Rajdhani", "Duronto", "Superfast", "Express"], p=[0.3, 0.3, 0.3, 0.1])
            elif cls in ("SL", "2S"):
                ttype = self.rng.choice(["Express", "Superfast", "Passenger"])
            elif cls == "CC":
                ttype = self.rng.choice(["Shatabdi", "Express", "Superfast"])
            else:
                ttype = self.rng.choice(["Rajdhani", "Duronto", "Superfast", "Express"])

            days_before = {"tomorrow": 1, "today": 0, "next week": 7}.get(constraints.get("when", "unspecified"), 7)
            is_peak = 1 if dep_hour in (17, 18, 19, 20) else 0
            dem_mult = 1.0 if is_peak else 0.88
            demand = float(np.clip(route["popularity"] * dem_mult + self.rng.normal(0, 0.03), 0.5, 1.0))
            fare_row = pd.DataFrame([{
                "route": route["code"], "travel_class": cls, "days_before": days_before,
                "demand_index": demand, "season_index": 0.2, "distance_km": route["distance_km"],
                "base_fare": float(CLASS_BASE_FARE[cls]),
            }])
            unit_fare = float(fare_model.predict(fare_row)[0])
            total_fare = round(unit_fare * max(1, passengers))

            speed = {"Rajdhani": 75, "Shatabdi": 80, "Duronto": 70, "Superfast": 65,
                     "Express": 55, "Passenger": 45, "MEMU": 50}[ttype]
            duration = round(route["distance_km"] / speed + float(self.rng.uniform(-0.5, 0.5)), 1)
            duration = max(1.0, duration)
            dep_hour = int(dep_hour)
            arr_hour = (dep_hour + duration) % 24
            delay_row = pd.DataFrame([{
                "train_type": ttype, "route": route["code"], "distance_km": route["distance_km"],
                "month": 10, "season_index": 0.2, "day_of_week": 2, "scheduled_dep_hour": dep_hour,
                "weather_severity": float(self.rng.uniform(0, 0.4)),
                "section_congestion": float(self.rng.uniform(0.2, 0.7)),
                "historical_avg_delay": (1 - TRAIN_TYPE_PUNCTUALITY[ttype]) * 70,
            }])
            delay = float(delay_model.predict(delay_row)[0])
            avail = float(np.clip(self.rng.beta(3, 2), 0.05, 1))
            arr_with_delay = round((dep_hour + duration + delay / 60) % 24, 1)
            candidates.append({
                "train_type": ttype, "class": cls, "fare": total_fare, "unit_fare": round(unit_fare),
                "duration_hr": duration,
                "dep_hour": dep_hour, "arr_hour": round(arr_hour, 1), "delay_min": round(delay, 1),
                "arr_with_delay": arr_with_delay,
                "punctuality": TRAIN_TYPE_PUNCTUALITY[ttype], "availability": round(avail, 3),
                "distance_km": route["distance_km"],
                "route_name": route["name"],
                "passengers": passengers,
            })

        # rank with the recommender model
        budget = constraints.get("budget") or 5000
        max_dur = constraints.get("max_duration_hr") or 24
        prio = (constraints.get("priority") or "comfort").lower()
        rows = []
        for c in candidates:
            rows.append({
                "budget": budget, "preferred_class": pcls,
                "max_duration_hr": max_dur, "priority": prio,
                "cand_class": c["class"], "cand_fare": c["fare"], "cand_duration_hr": c["duration_hr"],
                "cand_punctuality": c["punctuality"], "cand_availability": c["availability"],
                "cand_train_type": c["train_type"], "class_match": float(c["class"] == pcls),
                "within_budget": float(constraints.get("budget") is None or c["fare"] <= constraints["budget"]),
                "within_duration": float(constraints.get("max_duration_hr") is None or c["duration_hr"] <= max_dur),
            })
        scores = rec_model.predict_proba(pd.DataFrame(rows))[:, 1]
        for c, s in zip(candidates, scores):
            c["relevance_score"] = round(float(s), 4)

        return candidates

    def tool_search(self, constraints: dict, n: int = 40) -> list[dict]:
        candidates = self.generate_candidates(constraints, n=n)
        if not candidates:
            self._log("search_trains", constraints, "no route")
            return []
        self._log("search_trains", {"from": constraints.get("from"), "to": constraints.get("to"), "n": len(candidates)}, f"{len(candidates)} candidates")
        return candidates

    def tool_filter(self, candidates: list[dict], constraints: dict) -> list[dict]:
        out = []
        for c in candidates:
            if constraints.get("budget") and c["fare"] > constraints["budget"]:
                continue
            if constraints.get("arrive_by_hour") is not None and c["arr_with_delay"] > constraints["arrive_by_hour"]:
                continue
            out.append(c)
        self._log("filter_constraints", {"budget": constraints.get("budget"),
                                         "arrive_by": constraints.get("arrive_by_hour")},
                  f"{len(out)}/{len(candidates)} satisfy constraints")
        return out

    def tool_confirm_probability(self, candidate: dict, days_before: int = 1) -> float:
        model = self.platform.get_model("confirmation_predictor")
        wl = 0 if candidate["availability"] > 0.5 else int((1 - candidate["availability"]) * 80)
        row = pd.DataFrame([{
            "train_type": candidate["train_type"], "route": "PUNE-NDLS", "travel_class": candidate["class"],
            "quota": "GN", "booking_days_before": days_before, "month": 10, "season_index": 0.2,
            "is_peak_day": 0, "route_popularity": 0.8, "capacity": 350,
            "wl_position": wl, "rac_position": 0, "distance_km": candidate["distance_km"],
        }])
        p = float(model.predict_proba(row)[0, 1])
        return round(p, 3)

    def tool_book(self, itinerary: dict, approved: bool) -> dict:
        if not approved:
            self._log("book_ticket", {"itinerary": itinerary}, "blocked: awaiting user approval")
            return {"status": "pending_approval", "message": "Booking requires explicit user approval."}
        pnr = f"{self.rng.integers(10**9, 10**10)}"
        self._log("book_ticket", {"itinerary": itinerary}, f"booked PNR {pnr}")
        return {"status": "booked", "pnr": pnr, "itinerary": itinerary}

    # -- governance validation ----------------------------------------------
    def validate(self, constraints: dict, itinerary: dict | None) -> dict:
        checks = []
        if itinerary is None:
            return {"valid": False, "checks": [{"check": "itinerary_exists", "ok": False}]}
        if constraints["budget"] is not None:
            ok = itinerary["fare"] <= constraints["budget"]
            checks.append({"check": "within_budget", "ok": ok,
                           "detail": f"fare ₹{itinerary['fare']:,} vs budget ₹{constraints['budget']:,}"})
        if constraints["arrive_by_hour"] is not None:
            ok = itinerary["arr_with_delay"] <= constraints["arrive_by_hour"]
            arr_str = f"~{int(itinerary['arr_with_delay']):02d}:{int(round((itinerary['arr_with_delay'] % 1) * 60)):02d}"
            checks.append({"check": "meets_arrival_deadline", "ok": ok,
                           "detail": f"arrives {arr_str} vs deadline {constraints['arrive_by_hour']:02d}:00"})
        valid = all(c["ok"] for c in checks) if checks else True
        return {"valid": valid, "checks": checks}

    def _violation(self, c: dict, constraints: dict) -> float:
        v = 0.0
        if constraints.get("budget"):
            v += max(0.0, (c["fare"] - constraints["budget"]) / constraints["budget"])
        if constraints.get("arrive_by_hour") is not None:
            v += max(0.0, (c["arr_with_delay"] - constraints["arrive_by_hour"]) / 24.0)
        return v

    def _relaxations(self, constraints: dict, nearest: dict | None) -> list[str]:
        tips = []
        if nearest and constraints.get("budget") and nearest["fare"] > constraints["budget"]:
            tips.append(f"Raise the budget to about ₹{int(nearest['fare']):,} to include the cheapest option.")
        if nearest and constraints.get("arrive_by_hour") is not None and nearest["arr_with_delay"] > constraints["arrive_by_hour"]:
            arr_str = f"{int(nearest['arr_with_delay']):02d}:{int(round((nearest['arr_with_delay'] % 1) * 60)):02d}"
            tips.append(f"Relax the arrival deadline to about {arr_str}, or travel a day earlier.")
        if not tips:
            tips.append("Try a different date or class for more options.")
        return tips

    # -- agent run -----------------------------------------------------------
    def run(self, query: str, approve: bool = False) -> dict:
        self.last_steps = []
        constraints = self.parse(query)
        if not constraints["from"] or not constraints["to"]:
            return {"query": query, "constraints": constraints, "plan": None, "feasible": False,
                    "error": "Could not identify origin and destination.",
                    "steps": self.last_steps}
        candidates = self.tool_search(constraints, n=40)
        filtered = self.tool_filter(candidates, constraints)
        feasible = len(filtered) > 0

        prio = constraints.get("explicit_priority")
        if feasible:
            # If preferred class was specified and matching candidates exist, prioritize them unless explicit priority overrides
            if constraints.get("preferred_class") and not prio:
                cls_match = [c for c in filtered if c["class"] == constraints["preferred_class"]]
                if cls_match:
                    filtered = cls_match

            if prio == "price":
                filtered.sort(key=lambda c: (c["fare"], -c.get("relevance_score", 0)))
            elif prio == "speed":
                filtered.sort(key=lambda c: (c["duration_hr"], -c.get("relevance_score", 0)))
            elif prio == "punctuality":
                filtered.sort(key=lambda c: (-c["punctuality"], c["delay_min"], -c.get("relevance_score", 0)))
            else:
                # Fall back to recommender score
                filtered.sort(key=lambda c: c.get("relevance_score", 0), reverse=True)

            itinerary = dict(filtered[0])
            alternatives = filtered[1:4]
        else:
            # No option satisfies every hard constraint -> be honest, offer the closest.
            ranked = sorted(candidates, key=lambda c: (self._violation(c, constraints), c["fare"] if prio == "price" else -c.get("relevance_score", 0)))
            itinerary = dict(ranked[0]) if ranked else None
            alternatives = ranked[1:4]
            self._log("report_infeasible", {"constraints": constraints},
                      "no candidate met all hard constraints; proposing closest + relaxations")

        if itinerary:
            itinerary["confirmation_probability"] = self.tool_confirm_probability(itinerary)
        validation = self.validate(constraints, itinerary)
        # Booking is only allowed on a feasible, validated plan.
        if feasible and validation["valid"]:
            booking = self.tool_book(itinerary, approved=approve)
        else:
            booking = {"status": "blocked_infeasible",
                       "message": "No option satisfies all constraints; nothing to book. "
                                  "See suggested relaxations."}
            self._log("book_ticket", {}, "blocked: plan infeasible")
        return {
            "query": query,
            "constraints": constraints,
            "feasible": feasible,
            "n_candidates": len(candidates),
            "n_after_filter": len(filtered),
            "plan": itinerary,
            "alternatives": alternatives,
            "relaxations": [] if feasible else self._relaxations(constraints, itinerary),
            "validation": validation,
            "booking": booking,
            "steps": self.last_steps,
        }

