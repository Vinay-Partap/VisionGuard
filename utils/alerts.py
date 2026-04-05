# utils/alerts.py  (main branch — Windows, winsound)
import time
import winsound
import streamlit as st

ALERT_COOLDOWN = 3  # seconds


def should_alert():
    """Cooldown-based alert control."""
    now = time.time()
    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = 0
    if now - st.session_state.last_alert_time > ALERT_COOLDOWN:
        st.session_state.last_alert_time = now
        return True
    return False


def play_alert_sound():
    """Play alert using Windows winsound — bypasses browser autoplay restrictions."""
    try:
        winsound.PlaySound("assets/alert.wav", winsound.SND_FILENAME)
    except Exception:
        winsound.Beep(1000, 500)