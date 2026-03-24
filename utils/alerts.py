# utils/alerts.py — cross-platform, works on Windows (local) and Linux (cloud)
import time
import streamlit as st
import streamlit.components.v1 as components
import platform

ALERT_COOLDOWN = 3


def should_alert():
    now = time.time()
    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = 0
    if now - st.session_state.last_alert_time > ALERT_COOLDOWN:
        st.session_state.last_alert_time = now
        return True
    return False


def play_alert_sound():
    if platform.system() == "Windows":
        import winsound
        try:
            winsound.PlaySound("assets/alert.wav", winsound.SND_FILENAME)
        except Exception:
            winsound.Beep(1000, 500)
    else:
        # Cloud/Linux — use Web Audio API
        components.html("""
        <script>
        (function() {
            try {
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.type = 'sine';
                osc.frequency.setValueAtTime(880, ctx.currentTime);
                gain.gain.setValueAtTime(0.6, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.6);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.6);
            } catch(e) { console.warn('Beep failed:', e); }
        })();
        </script>
        """, height=0)