# utils/alerts.py
import time
import streamlit as st

ALERT_COOLDOWN = 3  # seconds

def should_alert():
    """Cooldown-based alert control"""
    now = time.time()

    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = 0

    if now - st.session_state.last_alert_time > ALERT_COOLDOWN:
        st.session_state.last_alert_time = now
        return True

    return False


def play_alert_sound():
    """Browser-safe alert sound"""
    if "alert_played" not in st.session_state:
        st.session_state.alert_played = False

    if not st.session_state.alert_played:
        st.audio("assets/alert.wav", format="audio/wav")
        st.session_state.alert_played = True
