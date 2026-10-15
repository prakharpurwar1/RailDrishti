"""AI Customer Support Assistant (multilingual, grounded, hallucination-aware).

A retrieval-grounded assistant: it classifies intent, retrieves the relevant
policy fact from the knowledge base, and composes a multilingual answer strictly
from retrieved evidence. Every answer is scored for groundedness so the
governance layer can monitor hallucination risk. An optional LLM hook can
replace the template composer without changing the monitoring contract.
"""
from __future__ import annotations

import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import DATA_DIR, SUPPORT_LANGUAGES
from ..data.knowledge_base import INTENT_KEYWORDS, KB_DOCS, KB_DOCS_ML, TEMPLATES
from ..governance.bias import hallucination_report


SCRIPT_RANGES = {
    "hi": ("\u0900", "\u097F"),   # Devanagari (also covers Marathi)
    "ta": ("\u0B80", "\u0BFF"),
    "te": ("\u0C00", "\u0C7F"),
    "bn": ("\u0980", "\u09FF"),
}


def detect_language(text: str) -> str:
    for lang, (lo, hi) in SCRIPT_RANGES.items():
        if any(lo <= ch <= hi for ch in text):
            return lang
    return "en"


class SupportAssistant:
    def __init__(self, pnr_store: pd.DataFrame | None = None):
        self.docs = KB_DOCS
        # Language-specific KBs; English lives in KB_DOCS, translations in KB_DOCS_ML.
        self.docs_by_lang = {"en": KB_DOCS, **KB_DOCS_ML}
        self.texts = [d["text"] for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self.pnr_store = pnr_store
        if self.pnr_store is None:
            try:
                df = pd.read_csv(DATA_DIR / "bookings.csv")
                self.pnr_store = df.set_index("pnr")
            except Exception:
                self.pnr_store = pd.DataFrame()

    # -- classification ------------------------------------------------------
    def classify_intent(self, text: str) -> tuple[str, float]:
        t = text.lower()
        scores = {}
        for intent, kws in INTENT_KEYWORDS.items():
            scores[intent] = sum(1 for k in kws if k in t)
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "fallback", 0.3
        return best, round(min(0.5 + 0.15 * scores[best], 0.98), 2)

    def retrieve(self, query: str, k: int = 2) -> list[dict]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        idx = np.argsort(sims)[::-1][:k]
        return [{"id": self.docs[i]["id"], "topic": self.docs[i]["topic"],
                 "text": self.docs[i]["text"], "score": round(float(sims[i]), 3)} for i in idx]

    INTENT_TOPIC = {
        "pnr_status": "pnr", "cancellation_policy": "cancellation", "tatkal_rules": "tatkal",
        "refund": "refund", "baggage": "baggage", "travel_documents": "travel_documents",
        "boarding": "boarding", "concession": "concession",
    }

    def retrieve_by_intent(self, intent: str, lang: str = "en") -> list[dict]:
        """Topic lookup in the requested language, so native-script queries and
        forced-language replies retrieve the right fact (lexical retrieval cannot
        match Devanagari/Tamil to English docs)."""
        topic = self.INTENT_TOPIC.get(intent, intent)
        source = self.docs_by_lang.get(lang) or self.docs
        docs = [d for d in source if d["topic"] == topic]
        return [{"id": d["id"], "topic": d["topic"], "text": d["text"], "score": 1.0} for d in docs]

    def _pnr_fact(self, pnr: str) -> str | None:
        if self.pnr_store is None or pnr not in self.pnr_store.index:
            return None
        row = self.pnr_store.loc[pnr]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return (f"PNR {pnr}: train {row['train_no']} ({row['route']}), class {row['travel_class']}, "
                f"current status {row['final_status']}, waitlist position {row['wl_position']}, "
                f"berth {row['berth_status'] or 'not allotted'}.")

    # -- generation ----------------------------------------------------------
    def _compose(self, intent: str, lang: str, evidence: list[dict]) -> str:
        tmpl = TEMPLATES.get(lang, TEMPLATES["en"]).get(intent) or TEMPLATES[lang]["fallback"]
        fact = evidence[0]["text"] if evidence else ""
        return tmpl.format(fact=fact) if fact else TEMPLATES[lang]["fallback"]

    def answer(self, query: str, language: str | None = None) -> dict:
        t0 = time.perf_counter()
        lang = language or detect_language(query)
        if lang not in TEMPLATES:
            lang = "en"
        intent, conf = self.classify_intent(query)
        evidence = []
        pnr_match = re.search(r"\b\d{10}\b", query)
        if intent == "pnr_status" and pnr_match:
            fact = self._pnr_fact(pnr_match.group(0))
            if fact:
                evidence = [{"id": "pnr-store", "topic": "pnr", "text": fact, "score": 1.0}]
        if not evidence:
            by_topic = self.retrieve_by_intent(intent, lang)
            evidence = by_topic if by_topic else self.retrieve(query)
        text = self._compose(intent, lang, evidence)
        if intent == "fallback" or not evidence:
            # A generic capability message makes no factual claim -> not a hallucination.
            hall = {"groundedness": 1.0, "hallucination": False, "risk": "n/a"}
        else:
            hall = hallucination_report(text, [e["text"] for e in evidence])
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "answer": text,
            "intent": intent,
            "confidence": conf,
            "language": lang,
            "evidence": evidence,
            "groundedness": hall["groundedness"],
            "hallucination": hall["hallucination"],
            "hallucination_risk": hall["risk"],
            "latency_ms": latency_ms,
        }
