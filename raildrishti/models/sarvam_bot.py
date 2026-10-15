"""Sarvam AI-powered Intelligent Railway Travel Companion.

Combines Sarvam's 105B Indic Conversational LLM, Bulbul TTS, and Saaras STT
with RailDrishti's real-time predictive models (confirmation, delay, fare, fraud, planner).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import ssl
import time
import urllib.request
from typing import Any

import certifi
import pandas as pd

from ..config import DATA_DIR

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "sk_vcjfv5mm_1oECOdM5FCp0SIgbSSBpzbpQ")
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# Mapping from common language codes to Sarvam BCP-47 tags & best speaker
LANG_SPEAKER_MAP = {
    "en": ("en-IN", "aditya"),
    "hi": ("hi-IN", "aditya"),
    "ta": ("ta-IN", "anand"),
    "te": ("te-IN", "vijay"),
    "bn": ("bn-IN", "manan"),
    "mr": ("mr-IN", "aditya"),
}


class SarvamRailBot:
    def __init__(self, platform=None, assistant=None, planner=None, api_key: str = SARVAM_API_KEY):
        self.platform = platform
        self.assistant = assistant
        self.planner = planner
        self.api_key = api_key

        # Lazy-loaded PNR store
        self.pnr_store = None
        try:
            pnr_csv = DATA_DIR / "bookings.csv"
            if pnr_csv.exists():
                df = pd.read_csv(pnr_csv)
                if "pnr" in df.columns:
                    self.pnr_store = df.set_index("pnr")
        except Exception:
            pass

    def _extract_pnr(self, query: str) -> str | None:
        m = re.search(r"\b\d{10}\b", query)
        return m.group(0) if m else None

    def _get_pnr_info(self, pnr: str) -> str | None:
        if self.pnr_store is None:
            return None
        try:
            if pnr in self.pnr_store.index:
                row = self.pnr_store.loc[pnr]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return (
                    f"PNR Record: {pnr}\n"
                    f"- Train: {row.get('train_no', 'N/A')} ({row.get('route', 'N/A')})\n"
                    f"- Travel Class: {row.get('travel_class', 'SL')}, Quota: {row.get('quota', 'GN')}\n"
                    f"- Current Status: {row.get('final_status', 'N/A')}\n"
                    f"- Waitlist Position: {row.get('wl_position', 0)}\n"
                    f"- Berth Status: {row.get('berth_status') or 'Not Allotted'}"
                )
        except Exception:
            pass
        return None

    def _get_confirmation_prediction(self, query: str) -> str | None:
        if not self.platform:
            return None
        q = query.lower()
        wl = re.search(r"(?:wl|waitlist|waiting list|waiting)[^0-9]{0,12}(\d{1,4})", q)
        if not wl:
            return None
        try:
            wl_pos = int(wl.group(1))
            days = re.search(r"(\d{1,3})\s*day", q)
            days_before = int(days.group(1)) if days else 18
            row = pd.DataFrame([{
                "train_type": "Express", "route": "PUNE-NDLS", "travel_class": "SL", "quota": "GN",
                "booking_days_before": days_before, "month": 10, "season_index": 0.2, "is_peak_day": 0,
                "route_popularity": 0.85, "capacity": 420, "wl_position": wl_pos,
                "rac_position": 0, "distance_km": 1515
            }])
            model = self.platform.get_model("confirmation_predictor")
            if model and hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(row)[0, 1])
                verdict = ("High chance of confirmation (Likely to confirm)" if prob >= 0.65
                           else "Moderate/Borderline chance" if prob >= 0.40
                           else "Low chance (likely to stay waitlisted)")
                return (
                    f"RailDrishti Confirmation Prediction Model:\n"
                    f"- Waitlist Position: WL {wl_pos}\n"
                    f"- Estimated Confirmation Probability: {prob * 100:.1f}%\n"
                    f"- Model Assessment: {verdict}"
                )
        except Exception:
            pass
        return None

    def _get_travel_plan(self, query: str) -> str | None:
        if not self.planner:
            return None
        q = query.lower()
        if any(h in q for h in ["plan a trip", "plan my journey", "travel from", "suggest train", "cheapest train", "reach by"]):
            try:
                res = self.planner.run(query)
                if res.get("feasible") and res.get("plan"):
                    p = res["plan"]
                    return (
                        f"RailDrishti Optimized Travel Planner Result:\n"
                        f"- Train Option: {p.get('train_type')} · Class {p.get('class')}\n"
                        f"- Estimated Fare: ₹{p.get('fare')}\n"
                        f"- Scheduled Arrival: {p.get('arr_with_delay', 0):.0f}:00 hrs\n"
                        f"- Historical Confirmation Chance: {p.get('confirmation_probability', 0)*100:.0f}%\n"
                        f"- Status: All passenger constraints satisfied"
                    )
            except Exception:
                pass
        return None

    def chat(self, query: str, history: list[dict] | None = None, language: str | None = None) -> dict:
        """Process a conversational query using Sarvam 105B Indic LLM + RailDrishti domain models."""
        t0 = time.perf_counter()
        query = query.strip()
        if not query:
            return {"answer": "Please ask a question about Indian Railways.", "model": "none", "latency_ms": 0}

        # Gather domain context
        context_parts = []
        pnr = self._extract_pnr(query)
        if pnr:
            pnr_info = self._get_pnr_info(pnr)
            if pnr_info:
                context_parts.append(pnr_info)

        conf_pred = self._get_confirmation_prediction(query)
        if conf_pred:
            context_parts.append(conf_pred)

        plan_info = self._get_travel_plan(query)
        if plan_info:
            context_parts.append(plan_info)

        # Knowledge base facts if assistant is available
        if self.assistant and hasattr(self.assistant, "retrieve"):
            try:
                retrieved = self.assistant.retrieve(query, k=2)
                for doc in retrieved:
                    if doc.get("score", 0) > 0.2:
                        context_parts.append(f"Official Policy Knowledge [{doc.get('topic')}]: {doc.get('text')}")
            except Exception:
                pass

        context_str = "\n\n".join(context_parts)

        # Build messages for Sarvam 105B
        system_content = (
            "You are RailDrishti AI (रेलदृष्टि एआई), the premier conversational AI assistant for Indian Railways.\n"
            "You talk like ChatGPT / Claude: helpful, conversational, insightful, natural, and polite.\n"
            "You possess deep expertise on Indian Railways: Tatkal booking timings, cancellation charges, refund rules, "
            "WL/RAC confirmation, PNR status, baggage limits, pet rules, senior citizen concessions, route suggestions, "
            "vande bharat / rajdhani trains, and IRCTC guidelines.\n"
            "Instructions:\n"
            "1. Answer clearly, directly, and thoroughly using Markdown (bullet points, bold text, tables where suitable).\n"
            "2. If domain facts or prediction numbers are provided in the Context below, use them accurately.\n"
            "3. If the user writes in Hindi or Hinglish, respond in natural, friendly Hindi/Hinglish. "
            "If they write in English or another language, reply in that language or as requested.\n"
            "4. Keep the tone warm, welcoming, and helpful, like an elite railway concierge."
        )

        if context_str:
            system_content += f"\n\n### Live RailDrishti Context & ML Predictions:\n{context_str}"

        messages = [{"role": "system", "content": system_content}]

        # Append recent history (up to last 6 turns)
        if history:
            for turn in history[-6:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})

        # Call Sarvam AI
        answer = None
        model_used = "sarvam-105b-conversations"
        try:
            req_body = {
                "model": "sarvam-105b-conversations",
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 1024,
            }
            req = urllib.request.Request(
                SARVAM_CHAT_URL,
                data=json.dumps(req_body).encode("utf-8"),
                headers={"api-subscription-key": self.api_key, "Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data["choices"][0]["message"]
                answer = choice.get("content")
        except Exception as e:
            # Fallback to local assistant if Sarvam network fails
            if self.assistant:
                try:
                    res = self.assistant.answer(query, language)
                    answer = res.get("answer")
                    model_used = "local-support-assistant"
                except Exception:
                    pass
            if not answer:
                answer = f"I apologize, I am temporarily having trouble reaching the knowledge service. ({str(e)})"
                model_used = "fallback"

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "answer": answer,
            "model": model_used,
            "latency_ms": latency_ms,
            "context_injected": bool(context_str),
            "pnr": pnr,
        }

    def tts(self, text: str, language: str = "en") -> str | None:
        """Generate high-quality voice audio using Sarvam Bulbul V3 TTS. Returns base64 data URL."""
        # Clean markdown formatting for smoother speech
        clean_text = re.sub(r"[*_#`\[\]]", "", text)
        clean_text = re.sub(r"https?://\S+", "", clean_text)
        clean_text = clean_text.strip()[:400]  # First paragraph/400 chars for prompt speech

        target_lang, speaker = LANG_SPEAKER_MAP.get(language, ("en-IN", "aditya"))
        payload = {
            "inputs": [clean_text],
            "target_language_code": target_lang,
            "speaker": speaker,
            "model": "bulbul:v3"
        }
        try:
            req = urllib.request.Request(
                SARVAM_TTS_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"api-subscription-key": self.api_key, "Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                audios = data.get("audios", [])
                if audios:
                    return f"data:audio/wav;base64,{audios[0]}"
        except Exception:
            pass
        return None

    def stt(self, audio_bytes: bytes, language: str = "en") -> str | None:
        """Transcribe audio using Sarvam Saaras V3 Speech-to-Text."""
        target_lang = LANG_SPEAKER_MAP.get(language, ("en-IN", "aditya"))[0]
        boundary = "----WebKitFormBoundaryRailDrishti7MA4YW"
        body = io.BytesIO()
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nsaaras:v3\r\n".encode("utf-8"))
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"language_code\"\r\n\r\n{target_lang}\r\n".encode("utf-8"))
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode("utf-8"))
        body.write(audio_bytes)
        body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        }
        try:
            req = urllib.request.Request(SARVAM_STT_URL, data=body.getvalue(), headers=headers)
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("transcript")
        except Exception:
            pass
        return None
