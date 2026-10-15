"""Agentic AI Supervisory Layer.

An autonomous agent that watches the other governed models: it observes health,
drift, data quality and bias, prioritises alerts, decides on a remediation, and
executes it through the lifecycle manager — then re-checks the result, closing
the loop. Every decision and its rationale is written to the audit trail.
"""
from __future__ import annotations

import time

import pandas as pd

from ..models.predictors import ALL_BUILDERS
from .platform import GovernancePlatform

SEVERITY_RANK = {"critical": 3, "high": 3, "drift": 3, "degraded": 2, "watch": 2, "medium": 2, "low": 1, "stable": 1}


class AgenticSupervisor:
    def __init__(self, platform: GovernancePlatform, criticality: dict | None = None):
        self.platform = platform
        self.audit = platform.audit
        # business criticality per model (higher = act more aggressively)
        self.criticality = criticality or {
            "confirmation_predictor": 0.9, "delay_predictor": 0.9, "fare_predictor": 0.7,
            "recommender": 0.6, "fraud_detector": 1.0, "support_assistant": 0.7,
            "travel_planner_agent": 0.6,
        }
        self.cycle_count = 0

    # -- observe -------------------------------------------------------------
    def observe(self) -> list[dict]:
        alerts = []
        for name in self.platform.registry.list_models():
            h = self.platform.health(name)
            sla_status = h["sla"]["status"]
            if sla_status != "healthy":
                alerts.append({
                    "model": name, "type": "sla_breach", "severity": sla_status,
                    "detail": f"SLA breached: {[c['sla'] for c in h['sla']['breached']]}",
                    "priority": SEVERITY_RANK[sla_status] * self.criticality.get(name, 0.5),
                })
            drift = self.platform.drift(name)
            if drift["severity"] != "stable":
                alerts.append({
                    "model": name, "type": "data_drift", "severity": drift["severity"],
                    "detail": f"{drift['n_drifted']} features drifted, max PSI {drift['max_psi']}",
                    "priority": SEVERITY_RANK[drift["severity"]] * self.criticality.get(name, 0.5),
                })
        alerts.sort(key=lambda a: a["priority"], reverse=True)
        return alerts

    # -- decide --------------------------------------------------------------
    def decide(self, alert: dict) -> dict:
        name = alert["model"]
        # Generative / agentic models cannot be auto-retrained -> escalate to a human.
        if name not in ALL_BUILDERS:
            return {"model": name, "action": "escalate_review",
                    "rationale": f"{name} is generative/agentic; auto-retrain not applicable. "
                                 f"Flagged for human review of prompts/tools/knowledge base.",
                    "alert": alert}
        # policy: match the alert type + severity to a remediation
        if alert["type"] == "sla_breach" and alert["severity"] == "critical":
            # severe regression -> prefer rollback if a prior good version exists
            has_prior = len(self.platform.registry.versions(name)) > 1
            action = "rollback" if has_prior else "retrain"
            rationale = ("Critical SLA breach and a prior version exists -> roll back to last "
                         "known-good, then reassess.") if has_prior else \
                        "Critical SLA breach, no prior version -> retrain immediately."
        elif alert["type"] == "data_drift" and alert["severity"] == "drift":
            action = "retrain"
            rationale = "Significant data drift -> retrain on the latest window and validate challenger."
        elif alert["type"] == "sla_breach":
            action = "retrain"
            rationale = "Degraded SLA -> retrain; promote only if the challenger improves."
        else:
            action = "monitor"
            rationale = "Early-warning drift -> increase monitoring cadence, no retrain yet."
        return {"model": name, "action": action, "rationale": rationale, "alert": alert}

    # -- act -----------------------------------------------------------------
    def act(self, decision: dict) -> dict:
        name, action = decision["model"], decision["action"]
        self.audit.record("agent_decision", actor="agentic_supervisor",
                          payload={"model": name, "action": action, "rationale": decision["rationale"],
                                   "alert": decision["alert"]})
        if action == "retrain":
            res = self.platform.retrain(name, reason=decision["alert"]["type"])
            # closed loop: verify the challenger actually fixed it
            if res.get("promoted"):
                after = self.platform.health(name)
                res["post_action_status"] = after["sla"]["status"]
                res["resolved"] = after["sla"]["status"] == "healthy"
                if not res["resolved"]:
                    # challenger promoted but still not healthy -> roll back and escalate
                    rb = self.platform.rollback(name, reason="post-retrain SLA still breached")
                    res["escalation"] = {"action": "rollback", **rb,
                                         "note": "escalated to human operator"}
            else:
                res["resolved"] = False
                res["escalation"] = {"action": "keep_incumbent",
                                     "note": "challenger rejected by validation gate"}
            return res
        if action == "rollback":
            out = self.platform.rollback(name, reason=decision["rationale"])
            after = self.platform.health(name)
            return {"model": name, "action": "rollback", **out,
                    "post_action_status": after["sla"]["status"],
                    "resolved": after["sla"]["status"] != "critical"}
        if action == "escalate_review":
            self.audit.record("escalate", actor="agentic_supervisor",
                              payload={"model": name, "reason": decision["rationale"]}, severity="warning")
            return {"model": name, "action": "escalate_review", "resolved": False,
                    "note": "ticket raised for human review"}
        return {"model": name, "action": "monitor", "resolved": False}

    # -- full cycle (closed loop) -------------------------------------------
    def run_cycle(self, max_actions: int = 4) -> dict:
        self.cycle_count += 1
        t0 = time.time()
        alerts = self.observe()
        decisions, actions = [], []
        acted = set()
        for alert in alerts:
            if len(actions) >= max_actions:
                break
            d = self.decide(alert)
            if d["action"] in ("monitor",) or d["model"] in acted:
                decisions.append(d)
                continue
            acted.add(d["model"])
            decisions.append(d)
            actions.append(self.act(d))
        result = {
            "cycle": self.cycle_count,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_alerts": len(alerts),
            "alerts": alerts,
            "decisions": decisions,
            "actions": actions,
            "resolved": sum(1 for a in actions if a.get("resolved")),
            "duration_s": round(time.time() - t0, 3),
        }
        self.audit.record("supervisor_cycle", actor="agentic_supervisor",
                          payload={"cycle": self.cycle_count, "n_alerts": len(alerts),
                                   "n_actions": len(actions), "resolved": result["resolved"]})
        return result
