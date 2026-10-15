#!/usr/bin/env bash
# RailDrishti - one-shot setup and pipeline runner.
set -e
cd "$(dirname "$0")"

echo "==> Installing dependencies"
pip install -q -r requirements.txt

echo "==> 1/4 Generating synthetic Indian Railways datasets"
python -m raildrishti.data.generator

echo "==> 2/4 Training & registering predictive models"
python scripts/train_all.py

echo "==> 3/4 Evaluating & registering generative + agentic models"
python scripts/evaluate_agents.py

echo "==> 4/4 Running the governance closed-loop demo"
python scripts/governance_demo.py

cat <<'EOF'

Done. To launch the platform:

  API        : uvicorn raildrishti.api.main:app --reload      (http://localhost:8000/docs)
  Dashboard  : streamlit run dashboard/app.py                 (http://localhost:8501)

EOF
