#!/bin/bash

# Navigate to script directory
cd "$(dirname "$0")"

# Check if .venv exists and activate it
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Warning: .venv directory not found. Trying to run without activation."
fi

# Cleanup existing process on port 8501
if lsof -i :8501 > /dev/null; then
    echo "Port 8501 is in use. Killing existing process..."
    fuser -k 8501/tcp
    sleep 1
fi

echo "Starting Streamlit Dashboard in BACKGROUND..."
echo "Access it at http://<your-server-ip>:8501"
echo "Logs are being written to streamlit.log"

# Run with nohup
nohup streamlit run app.py --server.port 8501 --server.address=0.0.0.0 > streamlit.log 2>&1 &

echo "Streamlit started with PID: $!"
echo "You can close this terminal now."
