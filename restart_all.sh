#!/bin/bash
# restart_all.sh — Restart engine and dashboard without pulling code.
# Use this after manual hotfixes on the VPS, or after a reboot.

set -euo pipefail

echo "Restarting pairs-engine..."
sudo systemctl restart engine

echo "Restarting dashboard..."
sudo systemctl restart dashboard

echo ""
sudo systemctl status engine dashboard --no-pager
