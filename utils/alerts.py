import time
import streamlit as st

# Cooldown time between alerts (seconds)
ALERT_COOLDOWN = 3

_last_alert_time = 0


def should_alert():
    """
    Returns True only if enough time has passed since the last alert.
    """
    global _last_alert_time
    current_time = time.time()

    if current_time - _last_alert_time > ALERT_COOLDOWN:
        _last_alert_time = current_time
        return True
    return False


def show_alert():
    """
    Displays a warning alert in Streamlit UI.
    """
    st.warning("🚨 ALERT: Pedestrian too close!")
