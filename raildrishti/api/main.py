"""FastAPI backend for RailDrishti.

Exposes the passenger-facing models and the full governance surface as REST
endpoints. Run with:
    uvicorn raildrishti.api.main:app --reload
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..governance.platform import GovernancePlatform
from ..governance.supervisor import AgenticSupervisor
from ..models.assistant import SupportAssistant
from ..models.planner import PlannerAgent
from ..models.sarvam_bot import SarvamRailBot

app = FastAPI(title="RailDrishti", version="1.0",
              description="AI-Powered Intelligent Railway Travel & Autonomous AI Governance Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_platform: GovernancePlatform | None = None
_supervisor: AgenticSupervisor | None = None
_assistant: SupportAssistant | None = None
_planner: PlannerAgent | None = None


def platform() -> GovernancePlatform:
    global _platform
    if _platform is None:
        _platform = GovernancePlatform()
    return _platform


def supervisor() -> AgenticSupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = AgenticSupervisor(platform())
    return _supervisor


def assistant() -> SupportAssistant:
    global _assistant
    if _assistant is None:
        _assistant = SupportAssistant()
    return _assistant


def planner() -> PlannerAgent:
    global _planner
    if _planner is None:
        _planner = PlannerAgent(platform())
    return _planner


_sarvam_bot: SarvamRailBot | None = None


def sarvam_bot() -> SarvamRailBot:
    global _sarvam_bot
    if _sarvam_bot is None:
        _sarvam_bot = SarvamRailBot(platform(), assistant(), planner())
    return _sarvam_bot


class ChatRequest(BaseModel):
    query: str
    language: str | None = None
    history: list[dict] | None = None


class TTSRequest(BaseModel):
    text: str
    language: str | None = "en"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class ConfirmationRequest(BaseModel):
    train_type: str = "Express"
    route: str = "PUNE-NDLS"
    travel_class: str = "SL"
    quota: str = "GN"
    booking_days_before: int = 15
    month: int = 10
    season_index: float = 0.2
    is_peak_day: int = 0
    route_popularity: float = 0.85
    capacity: int = 420
    wl_position: int = 25
    rac_position: int = 0
    distance_km: float = 1515


class DelayRequest(BaseModel):
    train_type: str = "Express"
    route: str = "PUNE-NDLS"
    distance_km: float = 1515
    month: int = 7
    season_index: float = 0.5
    day_of_week: int = 2
    scheduled_dep_hour: int = 18
    weather_severity: float = 0.4
    section_congestion: float = 0.5
    historical_avg_delay: float = 20


class FareRequest(BaseModel):
    route: str = "NDLS-BCT"
    travel_class: str = "3A"
    days_before: int = 10
    demand_index: float = 0.8
    season_index: float = 0.2
    base_fare: float = 1800
    distance_km: float = 1384


class FraudRequest(BaseModel):
    account_age_days: int = 400
    bookings_last_24h: int = 2
    distinct_pnrs: int = 3
    ip_count: int = 2
    device_count: int = 2
    failed_payments: int = 0
    avg_gap_sec: float = 3600
    payment_method: str = "UPI"





class PlanRequest(BaseModel):
    query: str
    approve: bool = False


class DriftRequest(BaseModel):
    model: str
    feature: str
    magnitude: float = 0.5


# ---------------------------------------------------------------------------
# Passenger-facing endpoints
# ---------------------------------------------------------------------------
def _predict(name: str, payload: BaseModel, proba: bool = False) -> dict[str, Any]:
    model = platform().get_model(name)
    row = pd.DataFrame([payload.model_dump()])
    if proba and model.spec.task in ("classification", "anomaly_classification"):
        p = float(model.predict_proba(row)[0, 1])
        return {"prediction": int(p >= model.threshold), "probability": round(p, 4),
                "explanation": platform().explain(name, row)}
    val = model.predict(row)[0]
    return {"prediction": float(val) if model.spec.task == "regression" else int(val),
            "explanation": platform().explain(name, row)}


@app.post("/predict/confirmation")
def predict_confirmation(req: ConfirmationRequest):
    out = _predict("confirmation_predictor", req, proba=True)
    p = out["probability"]
    out["confidence_score"] = p
    out["predicted_status"] = "CONFIRMED" if p >= 0.5 else ("RAC" if p >= 0.3 else "WAITLISTED")
    return out


@app.post("/predict/delay")
def predict_delay(req: DelayRequest):
    out = _predict("delay_predictor", req)
    out["unit"] = "minutes"
    out["disruption_risk"] = "high" if out["prediction"] > 60 else ("medium" if out["prediction"] > 30 else "low")
    return out


@app.post("/predict/fare")
def predict_fare(req: FareRequest):
    out = _predict("fare_predictor", req)
    out["currency"] = "INR"
    return out


@app.post("/predict/fraud")
def predict_fraud(req: FraudRequest):
    out = _predict("fraud_detector", req, proba=True)
    out["risk_score"] = out["probability"]
    out["action"] = "block_and_investigate" if out["probability"] >= 0.6 else "monitor"
    return out


@app.post("/assistant/chat")
@app.post("/api/chat")
def chat(req: ChatRequest):
    return sarvam_bot().chat(req.query, history=req.history, language=req.language)


@app.post("/api/tts")
def tts_endpoint(req: TTSRequest):
    audio_url = sarvam_bot().tts(req.text, language=req.language or "en")
    return {"audio_url": audio_url}


@app.post("/planner/plan")
def plan(req: PlanRequest):
    return planner().run(req.query, approve=req.approve)


# ---------------------------------------------------------------------------
# Governance endpoints
# ---------------------------------------------------------------------------
@app.get("/governance/report")
def governance_report():
    return platform().report()


@app.get("/governance/health/{model}")
def governance_health(model: str):
    try:
        return platform().health(model)
    except KeyError:
        raise HTTPException(404, f"unknown model '{model}'")


@app.get("/governance/drift/{model}")
def governance_drift(model: str):
    return platform().drift(model)


@app.get("/governance/bias/{model}")
def governance_bias(model: str):
    return platform().bias(model)


@app.get("/governance/root-cause/{model}")
def governance_rca(model: str, performance_drop: float = 0.0):
    return platform().root_cause(model, performance_drop=performance_drop)


@app.get("/governance/audit")
def governance_audit(limit: int = 100):
    ok, n = platform().audit.verify_chain()
    return {"chain_intact": ok, "n_records": n, "events": platform().audit.read(limit)}


@app.post("/governance/inject-drift")
def inject_drift(req: DriftRequest):
    platform().inject_drift(req.model, req.feature, magnitude=req.magnitude)
    return {"injected": True, "drift": platform().drift(req.model)}


@app.post("/governance/supervisor/cycle")
def supervisor_cycle():
    return supervisor().run_cycle()


@app.post("/governance/retrain/{model}")
def retrain(model: str, reason: str = "manual"):
    if model not in _platform_retrainable():
        raise HTTPException(400, f"{model} is not auto-retrainable")
    return platform().retrain(model, reason=reason)


def _platform_retrainable() -> set[str]:
    from ..models.predictors import ALL_BUILDERS
    return set(ALL_BUILDERS.keys())


@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    from .chat_ui import get_chat_page_html
    return HTMLResponse(get_chat_page_html())


@app.get("/api/status")
def api_status():
    return {"service": "RailDrishti", "docs": "/docs",
            "models": platform().registry.list_models()}
