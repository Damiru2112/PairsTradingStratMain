#!/bin/bash

# dashboard.service installer

SERVICE_NAME="dashboard.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_FILE="$CURRENT_DIR/$SERVICE_NAME"

echo "Installing $SERVICE_NAME..."

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then 
  echo "Please run this script with sudo:"
  echo "sudo ./install_service.sh"
  exit 1
fi

# Stop existing service if running
if systemctl is-active --quiet $SERVICE_NAME; then
    echo "Stopping existing service..."
    systemctl stop $SERVICE_NAME
fi

# Kill any existing streamlit process on port 8501 to free it up
echo "Checking for processes on port 8501..."
if lsof -i :8501 > /dev/null; then
    echo "Port 8501 is in use. Killing existing process..."
    fuser -k 8501/tcp
    sleep 2
fi

# Copy service file
echo "Copying service file to /etc/systemd/system/..."
cp "$SOURCE_FILE" "$SERVICE_PATH"
chmod 644 "$SERVICE_PATH"

# Reload daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable and start
echo "Enabling service to start on boot..."
systemctl enable $SERVICE_NAME

echo "Starting service..."
systemctl start $SERVICE_NAME

# Check status
echo "Service status:"
systemctl status $SERVICE_NAME --no-pager

echo ""
echo "Dashboard should now be accessible at http://<your-ip>:8501"
echo "To view logs: journalctl -u $SERVICE_NAME -f"
