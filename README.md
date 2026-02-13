# Pairs Trading System - Operations Guide

## 1. Dashboard Management (Streamlit)
The dashboard runs as a systemd service (`dashboard.service`), meaning it runs in the background and starts automatically on boot.

### **Access**
**URL**: http://170.64.216.29:8501

### **Restarting the Dashboard**
If the dashboard is stuck or you updated `app.py`:
```bash
sudo systemctl restart dashboard
```

### **Checking Status**
To see if it's running:
```bash
sudo systemctl status dashboard
```

### **Viewing Logs**
To see real-time logs (errors, print statements):
```bash
journalctl -u dashboard -f
```
*(Press `Ctrl+C` to exit logs)*

---

## 2. Trading Engine Management (Python)
The trading engine (`run_engine.py`) is currently run manually in the background using `nohup`.

### **Starting the Engine**
Use the start script to launch the engine (and check that it's running):
```bash
./start_all.sh
```
*Note: This script will verify if the engine is already running. It does NOT start the dashboard anymore since systemd handles that.*

### **Stopping the Engine**
To stop the trading engine:
```bash
pkill -f "run_engine.py"
```

### **Viewing Engine Logs**
```bash
tail -f engine_poly.log
```

---

## 3. Server Maintenance

### **System Restart**
If you reboot the VPS (`sudo reboot`):
1.  **Dashboard**: Will start AUTOMATICALLY. You don't need to do anything.
2.  **Trading Engine**: Will NOT start automatically. You must SSH in and run:
    ```bash
    cd PairsTrading
    ./start_all.sh
    ```

### **Updating Code**
If you pull new code from git or make changes:
1.  **Dashboard**: Restart to apply changes: `sudo systemctl restart dashboard`
2.  **Engine**: Restart if logic changed:
    ```bash
    pkill -f "run_engine.py"
    ./start_all.sh
    ```