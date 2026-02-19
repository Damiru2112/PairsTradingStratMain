#!/bin/bash
# deploy.sh — Pull latest code from GitHub and restart PairsBot services safely.
#
# Usage:
#   ./deploy.sh            — interactive (prompts during market hours)
#   ./deploy.sh --force    — skip market-hours prompt
#
# Prerequisites (one-time setup):
#   sudo cp engine.service /etc/systemd/system/
#   sudo systemctl daemon-reload && sudo systemctl enable engine
#   sudo visudo  # add NOPASSWD rules — see README/deploy docs

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

echo "=== PairsBot Deploy ==="
echo "Directory : $REPO_DIR"
echo "Time (UTC): $(date -u '+%Y-%m-%d %H:%M:%S')"
echo "Branch    : $(git rev-parse --abbrev-ref HEAD)"
echo "Current   : $(git rev-parse --short HEAD)"
echo ""

# --- 1. Safety: warn if US market is likely open ---------------------------
if [ "$FORCE" -eq 0 ]; then
    HOUR=$(TZ='America/New_York' date +%H)
    DOW=$(TZ='America/New_York' date +%u)  # 1=Mon … 7=Sun
    if [ "$DOW" -le 5 ] && [ "$HOUR" -ge 9 ] && [ "$HOUR" -lt 16 ]; then
        echo "WARNING: US market may be open ($(TZ='America/New_York' date '+%H:%M ET'))."
        echo "Restarting the engine will interrupt live positions."
        read -r -p "Continue? [y/N] " confirm
        [[ "$confirm" =~ ^[yY]$ ]] || { echo "Aborted."; exit 0; }
        echo ""
    fi
fi

# --- 2. Git pull (fast-forward only — refuses if local commits diverge) ----
echo "[1/4] Pulling latest changes..."
git pull --ff-only
echo "      Now at: $(git rev-parse --short HEAD)"
echo ""

# --- 3. Install / update Python dependencies --------------------------------
echo "[2/4] Installing dependencies..."
.venv/bin/pip install -q --require-virtualenv -r requirements.txt
echo "      Done."
echo ""

# --- 4. Import sanity check — catches syntax errors before restart ----------
echo "[3/4] Running import sanity check..."
.venv/bin/python - <<'EOF'
import sys, os
sys.path.insert(0, os.getcwd())
import db
import persist
import price_cache
import data_polygon
import strategy.live_trader
import analytics.betas
import analytics.zscore
print("      All critical imports OK.")
EOF
echo ""

# --- 5. Restart services ----------------------------------------------------
echo "[4/4] Restarting services..."
sudo systemctl restart engine
sleep 2
sudo systemctl restart dashboard
echo ""

echo "=== Deploy complete ==="
echo ""
sudo systemctl status engine dashboard --no-pager -l
