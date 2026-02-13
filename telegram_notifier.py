import os
import time
import logging
import requests
import threading
from datetime import datetime
# Load environment variables (manual parser to avoid dependencies)
def load_env(env_path=".env"):
    """Simple .env loader to avoid external dependencies like python-dotenv"""
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    # Strip quotes if present
                    value = value.strip(' "\'')
                    if key not in os.environ:
                        os.environ[key] = value
    except FileNotFoundError:
        pass  # It's okay if .env doesn't exist, we might rely on system env vars

load_env()

class TelegramNotifier:
    def __init__(self, token=None, chat_id=None, logger=None):
        """
        Initialize the Telegram Notifier.
        
        Args:
            token (str): BotFather token. Defaults to env var TELEGRAM_BOT_TOKEN.
            chat_id (str): Target Chat ID. Defaults to env var TELEGRAM_CHAT_ID.
            logger (logging.Logger): Logger instance. If None, creates a basic one.
        """
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        # Setup logging
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("TelegramNotifier")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

        if not self.token or not self.chat_id:
            self.logger.warning("Telegram Notifier initialized without Token or Chat ID. Notifications will fail.")

        self.last_sent = {}  # Stores timestamp of last message for keys
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_telegram(self, message: str, retries: int = 3):
        """
        Send a message to the configured Telegram chat.
        
        Args:
            message (str): The text message to send.
            retries (int): Number of retry attempts for network failures.
        """
        if not self.token or not self.chat_id:
            self.logger.error("Cannot send Telegram message: Missing Token or Chat ID.")
            return

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"  # Optional: allows bold/italic
        }

        for attempt in range(1, retries + 1):
            try:
                response = requests.post(self.base_url, json=payload, timeout=10)
                response.raise_for_status()
                # self.logger.info(f"Telegram message sent: {message[:50]}...")
                return  # Success
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Failed to send Telegram message (Attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(2)  # Wait a bit before retrying
                else:
                    self.logger.error("Max retries reached. Message not sent.")

    def notify(self, message: str):
        """
        Simple alias for send_telegram to fit requested API: notify("message")
        """
        # Run in a separate thread to not block the main trading loop
        # We want it lightweight, so just a daemon thread is fine for fire-and-forget
        thread = threading.Thread(target=self.send_telegram, args=(message,))
        thread.daemon = True
        thread.start()

    def notify_once(self, key: str, message: str, cooldown_minutes: int = 60):
        """
        Send a message only if it hasn't been sent with this 'key' 
        in the last 'cooldown_minutes'.
        
        Args:
            key (str): Unique identifier for this type of alert (e.g. "IBKR_DISCONNECT").
            message (str): The message content.
            cooldown_minutes (int): Minutes to wait before allowing this key again.
        """
        now = time.time()
        cooldown_seconds = cooldown_minutes * 60
        
        last_time = self.last_sent.get(key, 0)
        
        if now - last_time > cooldown_seconds:
            self.notify(message)
            self.last_sent[key] = now
        else:
            self.logger.info(f"Suppressed duplicate notification for key '{key}'.")

# Create a default global instance if needed for easy import
# Usage: from telegram_notifier import notifier; notifier.notify("Hello")
notifier = TelegramNotifier()
