# RailDrishti

**AI-Powered Intelligent Railway Travel & Autonomous AI Governance Platform**

RailDrishti is a working reference implementation of the case study. It has two layers that
watch each other: a **passenger-facing intelligence layer** (seven AI services that make a
railway journey smarter) and an **autonomous AI-governance layer** whose only job is to
monitor, validate, explain, heal and audit the models behind those services. The unifying
idea — *AI agents monitoring AI agents* — is what makes this more than another train app.

Everything runs locally with open-source libraries and **no paid APIs**. The data is a
realistic synthetic Indian Railways corpus, so every model learns genuine domain structure
(waiting-list conversion, monsoon delays, dynamic fares, bot behaviour) and can be retrained
on real feeds without changing a line of code.

---

## 1. Quick start

```bash
pip install -r requirements.txt

# Generate data, train every model, register it, run the governance demo
bash run.sh

# Then launch the two surfaces:
uvicorn raildrishti.api.main:app --reload      # REST API  -> http://localhost:8000/docs
streamlit run dashboard/app.py                 # Dashboard -> http://localhost:8501
```

The dashboard has a **Passenger** view (a conversational bot plus predictions and the planner)
and an **AI Governance Control Room** (health, drift, bias, RCA, the agentic supervisor, audit trail).

### Conversational bot — text and voice

The Passenger view opens on **RailBot**, a chat that routes each message to the right service:
policy/PNR/refund/Tatkal questions go to the grounded support assistant, "will WL 30 confirm?"
goes to the confirmation predictor, and "plan Pune to Delhi under 1500" goes to the travel-planner
agent. Every reply has a **🔊 Play** button (browser text-to-speech, no API key) and a compact
answer-quality panel (intent, groundedness, hallucination).

A **Reply language** selector forces answers in English, Hindi, Tamil, Telugu, Bengali or Marathi
(or auto-detects from the message script). The knowledge base is translated into all five Indian
languages (`raildrishti/data/kb_i18n/`), so a forced-language reply is fully in that language —
not an English fact wrapped in a translated sentence — and the voice playback follows the language.

Voice **input** uses `st.audio_input` (Streamlit ≥ 1.37) and a pluggable speech-to-text backend
that degrades gracefully to text-only:

```bash
pip install faster-whisper   # offline STT (auto-downloads a small model on first use)
```

Text-to-speech runs entirely in the browser via the Web Speech API; Indian-language voices depend
on what the operating system has installed.

---

## 2. Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          PASSENGER-FACING LAYER                             │
│  1 Ticket-Confirmation Predictor   2 Train Recommendation Engine            │
│  3 Delay & Disruption Predictor    4 AI Travel Planner Agent                │
│  5 Dynamic Fare Predictor          6 Multilingual Support Assistant         │
│  7 Fraud / Bot Detector                                                     │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │  every model registers, reports metrics,
                                │  exposes explanations & prediction events
┌───────────────────────────────▼────────────────────────────────────────────┐
│                   AUTONOMOUS AI GOVERNANCE LAYER (module 8)                 │
│                                                                             │
│  Model Registry & Versioning ── Model Health & SLA ── Data-Quality Gate      │
│  Data/Concept-Drift Detection ── Bias & Fairness ── Hallucination Monitor    │
│  Explainability (SHAP) ── Root-Cause Analysis ── Audit Trail (hash-chained)  │
│                                                                             │
│      ┌─────────────────── Agentic Supervisor ───────────────────┐           │
│      │  observe → prioritise → decide → act → re-verify (loop)  │           │
│      └──────────────────────────────────────────────────────────┘           │
└────────────────────────────────────────────────────────────────────────────┘
                                │
              FastAPI REST  ──►  Streamlit dashboard / any client
```

The governance layer is not a dashboard bolted on at the end: every predictive model is a
`GovernedModel` that carries its feature spec, SLA, protected attributes and drift features,
and the platform treats each one as a managed asset through a **single lifecycle** —
register → monitor → detect → diagnose → remediate → audit.

---

## 3. How each requirement is met

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | Ticket confirmation predictor | Gradient-boosted classifier; P(confirm) + confidence + SHAP explanation; self-retrains on drift |
| 2 | Train recommendation engine | Click/utility ranker over budget, duration, class, punctuality, availability; relevance scoring + bias checks |
| 3 | Delay & disruption prediction | Gradient-boosted regressor for arrival/departure delay + disruption risk; drift detection & auto-retrain |
| 4 | AI travel planner agent | NL understanding → tool loop (search → price → delay → confirm → rank → plan) → constraint validation → book on approval |
| 5 | Dynamic fare prediction | Gradient-boosted regressor for fare; booking-window advice; anomaly & accuracy monitoring |
| 6 | Customer-support assistant | Retrieval-grounded, multilingual (en/hi/ta/te/bn/mr) assistant; per-answer groundedness & hallucination scoring |
| 7 | Fraud / security / bot detection | Gradient-boosted classifier + isolation-forest anomaly scorer; recall-tuned threshold; explainable alerts |
| 8 | Centralized governance platform | Model health/SLA, data-quality gate, drift + concept drift, bias/fairness, hallucination monitor, RCA, lifecycle (retrain/rollback), audit trail, **agentic supervisor** |

**Non-functional**: scalability (stateless FastAPI + batch prediction), reliability
(challenger/rollback safety net), security (RBAC-ready API surface, risk scoring), compliance
(tamper-evident audit, explainability everywhere), performance (sub-second inference, p95 latency tracked per model).

---

## 4. Measured results

All figures from a held-out test split (75/25) of the synthetic corpus.

| Model | Task | Headline metrics |
|-------|------|------------------|
| Ticket confirmation | classification | AUC **0.986**, accuracy 0.96 |
| Delay prediction | regression | MAE **5.7 min**, R² 0.68 |
| Fare prediction | regression | R² **0.990**, MAPE 5.0% |
| Fraud detection | imbalanced classification | AUC **0.983**, recall 0.84 @ tuned threshold |
| Recommender | ranking | AUC 0.78, accuracy 0.73 |
| Support assistant | generation | groundedness **0.95**, hallucination rate **0.00** |
| Travel planner | agentic | task-success **1.00**, avg 4.2 tool calls |

**Closed-loop governance demo** (see `scripts/governance_demo.py`): injecting drift into
`wl_position` and `booking_days_before` (PSI 1.12) triggers drift detection → root-cause
analysis names the drifted high-importance feature → the supervisor autonomously retrains,
the challenger passes the validation gate and is promoted to `v2-auto`, and the audit hash
chain verifies intact.

---

## 5. Project structure

```
raildrishti/
├── raildrishti/
│   ├── config.py               # paths, domain constants, model manifest (SLAs)
│   ├── data/
│   │   ├── generator.py        # synthetic bookings / runs / fares / users / sessions
│   │   └── knowledge_base.py   # support KB + multilingual templates
│   ├── models/
│   │   ├── base.py             # GovernedModel + ModelSpec
│   │   ├── predictors.py       # the 5 predictive models
│   │   ├── assistant.py        # multilingual grounded assistant
│   │   ├── planner.py          # agentic travel planner
│   │   ├── railbot.py          # conversational bot: routes chat to assistant/planner/predictor
│   │   └── voice.py            # voice I/O: speech-to-text (pluggable) + browser text-to-speech
│   ├── governance/
│   │   ├── platform.py         # the central governance orchestrator
│   │   ├── registry.py         # versioning, promotion, rollback
│   │   ├── monitor.py          # health, SLA, latency
│   │   ├── drift.py            # PSI / KS / concept drift / behaviour change
│   │   ├── data_quality.py     # input validation gate
│   │   ├── bias.py             # fairness metrics + hallucination monitor
│   │   ├── explain.py          # SHAP explanations
│   │   ├── rca.py              # root-cause analysis + dependency tracing
│   │   ├── lifecycle.py        # retraining triggers, challenger gate
│   │   ├── audit.py            # tamper-evident hash-chained audit trail
│   │   └── supervisor.py       # agentic supervisory layer (closed loop)
│   └── api/main.py             # FastAPI REST surface
├── dashboard/
│   ├── app.py                  # Streamlit passenger (bot) + governance UI
│   └── styles.css              # Indian-Railways theme (navy + red on cream)
├── .streamlit/config.toml      # deterministic theme (light, IR palette)
├── scripts/                    # train_all · evaluate_agents · governance_demo
└── requirements.txt · run.sh
```

---

## 6. Design decisions

- **Synthetic-first data.** No public API exposes PNR/booking microdata, so the generator
  encodes the *causal structure* of railway demand (cancellations free berths → WL converts;
  monsoon and congestion drive delays; demand drives fare surges). Models trained here port
  to real feeds unchanged.
- **Governance as a first-class citizen.** The platform is the spine; every model implements
  the same contract (spec + metrics + explainability), so adding a model means registering it.
- **Safety in automation.** Auto-actions are gated: a challenger is promoted only if it meets
  its SLA *and* is no worse than the incumbent; generative/agentic models are escalated to a
  human rather than retrained blindly; the supervisor re-verifies after acting and rolls back
  if the fix fails.
- **Honest agent behaviour.** The planner reports infeasible requests explicitly (with
  suggested relaxations) instead of silently returning a plan that violates constraints.

## 7. Limitations & next steps

- The support assistant uses retrieval + templates (no LLM dependency). The `answer()`
  contract is LLM-ready: swap the composer and the groundedness/hallucination monitor still applies.
- Drift injection is a demo harness; in production the `current` window would be live traffic.
- Natural next steps: real IRCTC/data.gov.in feeds, streaming drift monitors (Evidently/Kafka),
  a vector store for the assistant, and role-based access control on the API.
