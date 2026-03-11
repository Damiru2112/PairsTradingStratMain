"""
Authentication module for the Pairs Trading Dashboard.

Roles:
  - "admin"  : Full access. Can view everything AND modify trading parameters,
                risk controls, and manual actions.
  - "guest"  : Read-only. Can view the dashboard, pair analysis, charts, etc.
                Cannot modify any parameters that affect live trading.

Credentials are stored as bcrypt hashes in a local JSON file (auth_users.json).
A default admin account is seeded on first run.
"""

from __future__ import annotations

import json
import hashlib
import hmac
import os
import secrets
from pathlib import Path
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AUTH_FILE = Path(__file__).parent / "auth_users.json"

# Roles
ROLE_ADMIN = "admin"
ROLE_GUEST = "guest"

# ---------------------------------------------------------------------------
# Password hashing (SHA-256 + salt, no extra dependency)
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Hash a password with a random salt. Returns (hash_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations=100_000)
    return h.hex(), salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against stored hash + salt."""
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations=100_000)
    return hmac.compare_digest(h.hex(), stored_hash)


# ---------------------------------------------------------------------------
# User store (JSON file)
# ---------------------------------------------------------------------------

def _load_users() -> dict:
    """Load user database from JSON file."""
    if not AUTH_FILE.exists():
        return {}
    with open(AUTH_FILE) as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    """Persist user database to JSON file."""
    with open(AUTH_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _ensure_default_admin() -> None:
    """Seed a default admin account if no users exist yet."""
    users = _load_users()
    if users:
        return
    # Default credentials - change after first login
    default_password = os.environ.get("DASHBOARD_ADMIN_PASSWORD", "admin")
    pw_hash, salt = _hash_password(default_password)
    users["admin"] = {
        "hash": pw_hash,
        "salt": salt,
        "role": ROLE_ADMIN,
    }
    _save_users(users)


def add_user(username: str, password: str, role: str = ROLE_GUEST) -> bool:
    """Add a new user. Returns False if username already exists."""
    users = _load_users()
    if username in users:
        return False
    pw_hash, salt = _hash_password(password)
    users[username] = {"hash": pw_hash, "salt": salt, "role": role}
    _save_users(users)
    return True


def change_password(username: str, new_password: str) -> bool:
    """Change password for an existing user."""
    users = _load_users()
    if username not in users:
        return False
    pw_hash, salt = _hash_password(new_password)
    users[username]["hash"] = pw_hash
    users[username]["salt"] = salt
    _save_users(users)
    return True


def authenticate(username: str, password: str) -> Optional[str]:
    """Authenticate a user. Returns their role on success, None on failure."""
    users = _load_users()
    user = users.get(username)
    if user is None:
        return None
    if _verify_password(password, user["hash"], user["salt"]):
        return user["role"]
    return None


# ---------------------------------------------------------------------------
# Streamlit session helpers
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize auth-related session state keys."""
    if "auth_role" not in st.session_state:
        st.session_state["auth_role"] = None
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None


def is_logged_in() -> bool:
    """Check if user has an active session (admin or guest)."""
    return st.session_state.get("auth_role") is not None


def is_admin() -> bool:
    """Check if current session has admin privileges."""
    return st.session_state.get("auth_role") == ROLE_ADMIN


def current_user() -> Optional[str]:
    """Return the current username, or None."""
    return st.session_state.get("auth_user")


def current_role() -> Optional[str]:
    """Return the current role, or None."""
    return st.session_state.get("auth_role")


def login(username: str, role: str) -> None:
    """Set session state for a logged-in user."""
    st.session_state["auth_user"] = username
    st.session_state["auth_role"] = role


def logout() -> None:
    """Clear session state."""
    st.session_state["auth_user"] = None
    st.session_state["auth_role"] = None


def login_as_guest() -> None:
    """Log in as a guest (no credentials needed)."""
    st.session_state["auth_user"] = "guest"
    st.session_state["auth_role"] = ROLE_GUEST


# ---------------------------------------------------------------------------
# Access request notifications
# ---------------------------------------------------------------------------
ADMIN_EMAIL = "ranasinghedamiru@gmail.com"


def _notify_access_request(name: str, email: str, description: str) -> None:
    """Send notification to admin when someone requests access."""
    msg = (
        f"Dashboard Access Request\n"
        f"Name: {name}\n"
        f"Email: {email}\n"
        f"Reason: {description or 'N/A'}"
    )

    # 1. Telegram notification (instant)
    try:
        from telegram_notifier import notifier
        notifier.notify(f"<b>New Access Request</b>\n\n"
                        f"<b>Name:</b> {name}\n"
                        f"<b>Email:</b> {email}\n"
                        f"<b>Reason:</b> {description or 'N/A'}")
    except Exception:
        pass  # Telegram not configured or unavailable

    # 2. Email notification
    try:
        import smtplib
        from email.mime.text import MIMEText
        import os as _os

        # Load .env if not already in environment
        from telegram_notifier import load_env
        load_env()

        smtp_user = _os.environ.get("SMTP_USER", "")
        smtp_pass = _os.environ.get("SMTP_PASSWORD", "")
        if smtp_user and smtp_pass:
            email_msg = MIMEText(
                f"Someone has requested access to the Pairs Trading Dashboard.\n\n"
                f"Name: {name}\n"
                f"Email: {email}\n"
                f"Reason: {description or 'N/A'}\n\n"
                f"Log in as admin to review, then email them their credentials."
            )
            email_msg["Subject"] = f"Dashboard Access Request from {name}"
            email_msg["From"] = smtp_user
            email_msg["To"] = ADMIN_EMAIL

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(email_msg)
    except Exception:
        pass  # Email not configured or failed


# ---------------------------------------------------------------------------
# Login page UI
# ---------------------------------------------------------------------------

def render_login_page() -> bool:
    """
    Render the login page. Returns True if user is now authenticated
    (either via login or guest), False if still on login screen.
    """
    _ensure_default_admin()
    init_session_state()

    if is_logged_in():
        return True

    # Center the login form
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0b0f14;
    }
    </style>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("## Pairs Trading Dashboard")
        st.markdown("---")

        tab_login, tab_guest, tab_request = st.tabs(
            ["Admin Login", "Guest Access", "Request Access"]
        )

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)

                if submitted:
                    role = authenticate(username, password)
                    if role:
                        login(username, role)
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        with tab_guest:
            st.info("Guest access allows you to view the dashboard and pair analysis, "
                    "but you cannot modify trading parameters or risk controls.")
            if st.button("Continue as Guest", use_container_width=True):
                login_as_guest()
                st.rerun()

        with tab_request:
            st.info("Request an account to get full access. "
                    "Your request will be reviewed and credentials sent to your email.")
            with st.form("access_request_form"):
                req_name = st.text_input("Full Name")
                req_email = st.text_input("Email Address")
                req_desc = st.text_area(
                    "Why do you need access?",
                    placeholder="Brief description about yourself and why you need access..."
                )
                req_submit = st.form_submit_button("Submit Request", use_container_width=True)

                if req_submit:
                    if not req_name or not req_email:
                        st.error("Name and email are required.")
                    elif "@" not in req_email:
                        st.error("Please enter a valid email address.")
                    else:
                        try:
                            import db as _db
                            _con = _db.connect_db()
                            _db.init_db(_con)
                            _db.add_access_request(_con, req_name, req_email, req_desc)

                            # Notify admin via email + Telegram
                            _notify_access_request(req_name, req_email, req_desc)

                            st.success(
                                "Request submitted! You will receive an email "
                                "with your login credentials once approved."
                            )
                        except Exception as e:
                            st.error(f"Failed to submit request: {e}")

    return False


# ---------------------------------------------------------------------------
# Password rotation timestamp
# ---------------------------------------------------------------------------
ROTATION_TIMESTAMP_FILE = Path(__file__).parent / ".last_pw_rotation"


def get_time_since_pw_rotation() -> Optional[str]:
    """Return a human-readable string like '3d 5h' since last admin password rotation."""
    if not ROTATION_TIMESTAMP_FILE.exists():
        return None
    try:
        from datetime import datetime, timezone, timedelta
        ts = datetime.fromisoformat(ROTATION_TIMESTAMP_FILE.read_text().strip())
        delta = datetime.now(timezone.utc) - ts
        days = delta.days
        hours = delta.seconds // 3600
        if days > 0:
            return f"{days}d {hours}h"
        if hours > 0:
            return f"{hours}h {(delta.seconds % 3600) // 60}m"
        mins = delta.seconds // 60
        return f"{mins}m"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar auth status (call from main app after login)
# ---------------------------------------------------------------------------

def render_auth_sidebar() -> None:
    """Show current user/role in sidebar with logout button."""
    with st.sidebar:
        role_label = "Admin" if is_admin() else "Guest"
        st.markdown(f"**{current_user()}** ({role_label})")
        if st.button("Logout", key="auth_logout", use_container_width=True):
            logout()
            st.rerun()

        # Admin: show pending access requests count
        if is_admin():
            try:
                import db as _db
                _con = _db.connect_db()
                _db.init_db(_con)
                pending = _db.get_access_requests(_con, status="PENDING")
                if not pending.empty:
                    st.warning(f"{len(pending)} pending access request(s)")
                    with st.expander("View Requests"):
                        for _, row in pending.iterrows():
                            st.markdown(f"**{row['name']}** ({row['email']})")
                            st.caption(row.get("description", ""))
                            st.caption(f"Submitted: {row['created_at']}")
                            st.markdown("---")
            except Exception:
                pass
