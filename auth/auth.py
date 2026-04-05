# auth/auth.py
import json
import hashlib
import os
import streamlit as st
from datetime import datetime

USERS_FILE = "auth/users.json"

DEFAULT_USERS = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "display_name": "Administrator",
    },
    "viewer": {
        "password": hashlib.sha256("viewer123".encode()).hexdigest(),
        "role": "viewer",
        "display_name": "Viewer",
    },
}


def _load_users():
    if not os.path.exists(USERS_FILE):
        os.makedirs("auth", exist_ok=True)
        with open(USERS_FILE, "w") as f:
            json.dump(DEFAULT_USERS, f, indent=2)
        return DEFAULT_USERS
    with open(USERS_FILE) as f:
        return json.load(f)


def _save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def authenticate(username, password):
    users = _load_users()
    user = users.get(username.strip().lower())
    if user and user["password"] == _hash(password):
        return {"username": username, "role": user["role"], "display_name": user["display_name"]}
    return None


def add_user(username, password, role, display_name):
    users = _load_users()
    users[username.lower()] = {
        "password": _hash(password),
        "role": role,
        "display_name": display_name,
        "created_at": datetime.now().strftime("%Y-%m-%d"),
    }
    _save_users(users)


def delete_user(username):
    users = _load_users()
    users.pop(username.lower(), None)
    _save_users(users)


def list_users():
    users = _load_users()
    return [{"username": k, "role": v["role"], "display_name": v["display_name"]}
            for k, v in users.items()]


# ── Streamlit session helpers ──────────────────────────────────────────────────

def is_logged_in():
    return st.session_state.get("logged_in", False)


def current_user():
    return st.session_state.get("user", {})


def is_admin():
    return current_user().get("role") == "admin"


def logout():
    st.session_state.pop("logged_in", None)
    st.session_state.pop("user", None)
    st.rerun()


def show_login_page():
    st.markdown("""
    <style>
    .login-wrap {
        max-width: 400px; margin: 8vh auto 0;
        background: #1e1e2e; border: 1px solid #2e2e4e;
        border-radius: 14px; padding: 2.5rem 2rem;
    }
    .login-title {
        font-size: 1.8rem; font-weight: 700;
        color: #a78bfa; text-align: center;
        letter-spacing: 0.05em; margin-bottom: 0.3rem;
    }
    .login-sub {
        text-align: center; color: #555; font-size: 0.78rem;
        margin-bottom: 1.5rem; letter-spacing: 0.12em;
    }
    .creds-box {
        font-size: 0.72rem; color: #444; background: #111;
        border: 1px solid #2e2e4e; border-radius: 6px;
        padding: 0.5rem 0.8rem; margin-top: 1rem; line-height: 1.9;
    }
    </style>
    <div class="login-wrap">
      <div class="login-title">👁️ VisionGuard AI</div>
      <div class="login-sub">REAL-TIME TRAFFIC INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 3, 1])
    with col:
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("🔐 Login", use_container_width=True, type="primary"):
            user = authenticate(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.markdown("""
        <div class="creds-box">
        🔑 Default credentials<br>
        &nbsp;&nbsp;<b>admin</b> / admin123 → Full access<br>
        &nbsp;&nbsp;<b>viewer</b> / viewer123 → Read-only
        </div>
        """, unsafe_allow_html=True)