#!/bin/bash

# Navigate to directory
cd "$(dirname "$0")"

# Activate venv
source .venv/bin/activate

# 1. Start Engine (Polygon 15m mode)
# 1. Start Engine (Polygon 15m mode)
# echo "Starting Engine (15m)..."
# nohup python -u run_engine.py --mode=15m --engine-id=POLYGON_MAIN > engine_poly.log 2>&1 &
# echo "Engine PID: $!"

# 1. Start Engine (1-Minute Mode)
echo "Starting Engine (1m)..."
nohup python -u run_engine_1m.py --engine-id=US_1M > engine_1m.log 2>&1 &
echo "Engine PID: $!"

# 2. Start Streamlit (Managed by systemd now)
# echo "Starting Dashboard..."
# nohup streamlit run app.py \
#     --server.port 8501 \
#     --server.headless=true \
#     --server.address=0.0.0.0 \
#     > streamlit.log 2>&1 &
# echo "Streamlit PID: $!"

echo "All started. Logs: engine_poly.log, streamlit.log"
