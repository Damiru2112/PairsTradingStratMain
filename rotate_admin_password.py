#!/usr/bin/env python3
"""
Admin password rotation script.

Run manually whenever you want to rotate the 'admin' account password.
It will:
  1. Generate a new random password
  2. Update local auth_users.json
  3. SSH into the VPS and update the remote auth_users.json
  4. Update ~/Desktop/dashboard_credentials.py
  5. Record the rotation timestamp (for the dashboard timer)

Usage:
    python3 rotate_admin_password.py
"""

from __future__ import annotations

import json
import secrets
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).parent))
from auth import change_password, _hash_password, _load_users, AUTH_FILE

# --- Config ---
VPS_HOST = "170.64.216.29"
VPS_USER = "trader"
VPS_PROJECT = "/home/trader/PairsTrading"
VPS_AUTH_FILE = f"{VPS_PROJECT}/auth_users.json"
CREDENTIALS_FILE = Path.home() / "Desktop" / "dashboard_credentials.py"
ROTATION_TIMESTAMP_FILE = Path(__file__).parent / ".last_pw_rotation"


def generate_password(length: int = 16) -> str:
    """Generate a strong random password."""
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


def update_vps(new_users_json: str) -> bool:
    """SSH into VPS and overwrite the remote auth_users.json."""
    try:
        # Write the JSON via stdin to avoid shell escaping issues
        result = subprocess.run(
            ["ssh", f"{VPS_USER}@{VPS_HOST}",
             f"cat > {VPS_AUTH_FILE}"],
            input=new_users_json,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"  SSH error: {result.stderr.strip()}")
            return False
        print("  VPS auth_users.json updated.")
        return True
    except subprocess.TimeoutExpired:
        print("  SSH timed out after 30s.")
        return False
    except FileNotFoundError:
        print("  SSH not found. Is OpenSSH installed?")
        return False


def update_credentials_file(new_password: str) -> None:
    """Rewrite the desktop credentials file with the new admin password."""
    today = datetime.now().strftime("%Y-%m-%d")
    content = f'''"""
Pairs Trading Dashboard - Login Credentials
=============================================
Last updated: {today}

KEEP THIS FILE PRIVATE. Do not share or commit to git.
The 'admin' account password is rotated by running:
    python3 rotate_admin_password.py
"""

credentials = {{
    "DRan": {{
        "password": "AmysAngels",
        "role": "admin",
        "note": "Permanent password",
    }},
    "FBurg": {{
        "password": "AmysAngels",
        "role": "admin",
        "note": "Permanent password",
    }},
    "admin": {{
        "password": "{new_password}",
        "role": "admin",
        "note": "Last changed {today}",
    }},
}}

if __name__ == "__main__":
    print("=== Dashboard Credentials ===")
    print()
    for user, info in credentials.items():
        print(f"  Username: {{user}}")
        print(f"  Password: {{info[\'password\']}}")
        print(f"  Role:     {{info[\'role\']}}")
        print(f"  Note:     {{info[\'note\']}}")
        print()
'''
    CREDENTIALS_FILE.write_text(content)


def save_rotation_timestamp() -> None:
    """Save the current UTC timestamp so the dashboard can show 'time since rotation'."""
    ROTATION_TIMESTAMP_FILE.write_text(datetime.now(timezone.utc).isoformat())


def main() -> None:
    users = _load_users()
    if "admin" not in users:
        print("ERROR: 'admin' account not found in auth_users.json")
        sys.exit(1)

    new_pw = generate_password()
    print(f"Generated new admin password: {new_pw}")

    # 1. Update local auth hash
    if not change_password("admin", new_pw):
        print("ERROR: Failed to update local auth_users.json")
        sys.exit(1)
    print("  Local auth_users.json updated.")

    # 2. Push updated auth_users.json to VPS
    updated_users = json.dumps(_load_users(), indent=2)
    if not update_vps(updated_users):
        print("WARNING: VPS update failed. You may need to manually sync.")

    # 3. Update desktop credentials file
    update_credentials_file(new_pw)
    print(f"  Desktop credentials updated: {CREDENTIALS_FILE}")

    # 4. Record rotation timestamp
    save_rotation_timestamp()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Admin password rotated successfully.")


if __name__ == "__main__":
    main()
