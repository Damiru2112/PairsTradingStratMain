#!/usr/bin/env python3
"""
VPS-side admin password rotation (run via cron every 2 weeks).

  1. Generate a new random password
  2. Update auth_users.json
  3. Commit & push .admin_password to GitHub
  4. Record the rotation timestamp
  5. Send confirmation via Telegram (no password in message)
"""

from __future__ import annotations

import secrets
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).parent))
from auth import change_password, _load_users

PROJECT_DIR = Path(__file__).parent
ROTATION_TIMESTAMP_FILE = PROJECT_DIR / ".last_pw_rotation"
ADMIN_PW_FILE = PROJECT_DIR / ".admin_password"


def generate_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    pw = [
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.digits),
        secrets.choice("!@#$%^&*()-_=+"),
    ]
    pw += [secrets.choice(alphabet) for _ in range(length - 4)]
    pw_list = list(pw)
    secrets.SystemRandom().shuffle(pw_list)
    return "".join(pw_list)


def push_password_to_github(new_pw: str) -> bool:
    """Write password to .admin_password, commit, and push."""
    ADMIN_PW_FILE.write_text(new_pw)
    try:
        subprocess.run(["git", "add", ".admin_password"],
                       cwd=PROJECT_DIR, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Rotate admin password"],
            cwd=PROJECT_DIR, check=True, capture_output=True,
        )
        subprocess.run(["git", "push"], cwd=PROJECT_DIR, check=True,
                        capture_output=True, timeout=30)
        print("  .admin_password pushed to GitHub.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: Git push failed: {e.stderr.decode().strip()}")
        return False
    except subprocess.TimeoutExpired:
        print("  WARNING: Git push timed out.")
        return False


def main() -> None:
    users = _load_users()
    if "admin" not in users:
        print("ERROR: 'admin' account not found in auth_users.json")
        sys.exit(1)

    new_pw = generate_password()

    if not change_password("admin", new_pw):
        print("ERROR: Failed to update auth_users.json")
        sys.exit(1)
    print("  auth_users.json updated.")

    # Push password to GitHub
    push_password_to_github(new_pw)

    # Record rotation timestamp
    ROTATION_TIMESTAMP_FILE.write_text(datetime.now(timezone.utc).isoformat())

    # Send confirmation via Telegram (no password)
    try:
        from telegram_notifier import notifier
        notifier.send_telegram(
            f"<b>Admin Password Rotated</b>\n\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Open credentials file on Desktop to see the new password."
        )
        print("  Telegram confirmation sent.")
    except Exception as e:
        print(f"  WARNING: Telegram notification failed: {e}")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Admin password rotated.")


if __name__ == "__main__":
    main()
