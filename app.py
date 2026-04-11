# app.py  (main branch — local Windows, cv2 + winsound)
import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime
from io import BytesIO

# ── Auth ──────────────────────────────────────────────────────────────────────
from auth.auth import (
    is_logged_in, current_user, is_admin, show_login_page, logout,
    add_user, delete_user, list_users,
)

# ── Original imports ──────────────────────────────────────────────────────────
from detector.yolo_detector import (
    detect_objects, reset_tracker, get_speed_estimator,
)
from utils.alerts import should_alert, play_alert_sound
from utils.summary import init_summary, reset_summary

# ── New feature imports ───────────────────────────────────────────────────────
from utils.heatmap import HeatmapAccumulator
from utils.danger_zones import DangerZoneManager
from reports.road_report import (
    start_session, end_session, generate_report,
    get_all_sessions_df, get_road_names, export_report_txt,
)

st.set_page_config(page_title="VisionGuard AI", layout="wide", page_icon="👁️")

# ─────────────────────────────────────────
# CUSTOM CSS  (original + new additions)
# ─────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e; border-radius: 12px;
        padding: 16px 20px; text-align: center;
        border: 1px solid #2e2e4e;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #a78bfa; }
    .metric-label { font-size: 0.8rem; color: #888; margin-top: 4px; }
    .alert-high   { background:#ff4b4b22; border-left: 4px solid #ff4b4b; padding: 8px 14px; border-radius: 4px; }
    .alert-medium { background:#ffa50022; border-left: 4px solid #ffa500; padding: 8px 14px; border-radius: 4px; }
    .risk-high    { color: #ff4b4b; font-weight: 600; }
    .risk-medium  { color: #ffa500; font-weight: 600; }
    .risk-low     { color: #21c55d; font-weight: 600; }
    div[data-testid="stSidebar"] { background: #111827; }
    .road-safe    { background:rgba(33,197,93,.08); border-left:4px solid #21c55d; padding:.6rem 1rem; border-radius:4px; margin:.4rem 0; }
    .road-caution { background:rgba(255,165,0,.08);  border-left:4px solid #ffa500; padding:.6rem 1rem; border-radius:4px; margin:.4rem 0; }
    .road-avoid   { background:rgba(255,75,75,.08);  border-left:4px solid #ff4b4b; padding:.6rem 1rem; border-radius:4px; margin:.4rem 0; }
    .user-badge   { font-size:.75rem; color:#888; background:#1e1e2e; border:1px solid #2e2e4e; border-radius:4px; padding:.2rem .6rem; display:inline-block; margin-bottom:.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# AUTH GATE
# ─────────────────────────────────────────
if not is_logged_in():
    show_login_page()
    st.stop()

user = current_user()

# ─────────────────────────────────────────
# SIDEBAR  (original + road name + new nav)
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/eye.png", width=60)
    st.title("VisionGuard AI")
    st.caption("Real-Time Detection System")
    st.markdown(f'<div class="user-badge">👤 {user["display_name"]} | {user["role"].upper()}</div>',
                unsafe_allow_html=True)
    st.divider()

    # ── Page navigation ───────────────────────────────────────────────────────
    pages = ["🎯 Detection", "🌡️ Heatmap", "⚠️ Danger Zones", "📊 Reports", "📋 Sessions"]
    if is_admin():
        pages.append("🔧 Admin")
    page = st.radio("Navigation", pages, label_visibility="collapsed")
    st.divider()

    # ── Detection settings (shown on Detection page) ──────────────────────────
    if "Detection" in page:
        st.subheader("⚙️ Detection Settings")

        road_name = st.text_input(
            "📍 Road / Location",
            value=st.session_state.get("road_name", ""),
            placeholder="e.g. NH-48 Sector 15",
        )
        st.session_state.road_name = road_name

        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.1, max_value=0.9, value=0.4, step=0.05,
        )
        proximity_threshold = st.slider(
            "Alert distance (meters)",
            min_value=1.0, max_value=20.0, value=7.0, step=0.5,
        )
        sound_enabled = st.toggle("🔊 Alert sound", value=True)
        show_heatmap  = st.toggle("🌡️ Heatmap overlay", value=False)
        show_zones    = st.toggle("⚠️ Danger zones overlay", value=True)
        speed_limit   = st.slider("🚗 Speed limit (km/h)", 10, 120, 30, 5)
        pixels_per_m  = st.slider("📏 Pixels per metre", 10, 100, 40, 5,
                                   help="Calibration for speed estimation")

        st.divider()
        st.subheader("🎨 Risk legend")
        st.markdown('<span class="risk-high">● HIGH</span> &nbsp; < 4m', unsafe_allow_html=True)
        st.markdown('<span class="risk-medium">● MEDIUM</span> &nbsp; 4–7m', unsafe_allow_html=True)
        st.markdown('<span class="risk-low">● LOW</span> &nbsp; > 7m', unsafe_allow_html=True)

    st.divider()
    if st.button("🔓 Logout", use_container_width=True):
        logout()
    st.caption("VisionGuard AI v2.0")

# ─────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────
if "summary"         not in st.session_state: st.session_state.summary         = init_summary()
if "fps_list"        not in st.session_state: st.session_state.fps_list        = []
if "last_frame_time" not in st.session_state: st.session_state.last_frame_time = time.time()
if "heatmap_acc"     not in st.session_state: st.session_state.heatmap_acc     = HeatmapAccumulator(decay=0.97, radius=50)
if "zone_mgr"        not in st.session_state: st.session_state.zone_mgr        = DangerZoneManager()
if "active_session"  not in st.session_state: st.session_state.active_session  = None

# ─────────────────────────────────────────
# SHARED HELPERS  (original functions kept exactly)
# ─────────────────────────────────────────
def compute_fps():
    now = time.time()
    elapsed = now - st.session_state.last_frame_time
    st.session_state.last_frame_time = now
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    st.session_state.fps_list.append(fps)
    if len(st.session_state.fps_list) > 30:
        st.session_state.fps_list.pop(0)
    return round(sum(st.session_state.fps_list) / len(st.session_state.fps_list), 1)


def update_metrics(summary, fps=0.0):
    pedestrian_metric.metric("🚶 Pedestrians", summary["pedestrians"])
    vehicle_metric.metric("🚗 Vehicles",       summary["vehicles"])
    alert_metric.metric("🚨 Alerts",           summary["alerts"])
    fps_metric.metric("⚡ FPS",                f"{fps:.1f}")
    total_metric.metric("📦 Total",            summary["total"])


def trigger_alert(summary):
    alert_box.markdown(
        '<div class="alert-high">🚨 <b>Pedestrian too close! Immediate danger.</b></div>',
        unsafe_allow_html=True,
    )
    if sound_enabled:
        play_alert_sound()


def export_csv(summary):
    if not summary["detection_log"]:
        return None
    df = pd.DataFrame(summary["detection_log"])
    return df.to_csv(index=False).encode("utf-8")


def export_pdf_txt(summary):
    lines = [
        "=" * 50, "       VISIONGUARD AI — DETECTION REPORT", "=" * 50,
        f"Session start : {summary['start_time']}",
        f"Report time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "", "SUMMARY", "-" * 30,
        f"  Pedestrians detected : {summary['pedestrians']}",
        f"  Vehicles detected    : {summary['vehicles']}",
        f"  Total detections     : {summary['total']}",
        f"  Alerts triggered     : {summary['alerts']}",
        "", "DETECTION LOG", "-" * 30,
    ]
    for entry in summary["detection_log"]:
        lines.append(
            f"  [{entry['time']}]  {entry['type']:12s}  "
            f"dist={entry['distance_m']}m  risk={entry['risk'].upper()}"
        )
    lines += ["", "=" * 50, "  Generated by VisionGuard AI v2.0", "=" * 50]
    return "\n".join(lines).encode("utf-8")


def _show_log_and_export(summary):
    """Reusable detection log + export block used in all input modes."""
    st.divider()
    st.subheader("📋 Detection history log")
    if summary["detection_log"]:
        df = pd.DataFrame(summary["detection_log"])
        df.columns = ["Time", "Type", "Distance (m)", "Risk"]
        def color_risk(val):
            return {"high": "color: #ff4b4b", "medium": "color: #ffa500",
                    "low": "color: #21c55d"}.get(val.lower(), "")
        st.dataframe(df.style.map(color_risk, subset=["Risk"]),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No detections yet.")

    st.divider()
    st.subheader("📥 Export report")
    col1, col2 = st.columns(2)
    with col1:
        csv_data = export_csv(summary)
        if csv_data:
            st.download_button("⬇️ Download CSV", data=csv_data,
                               file_name=f"visionguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv", use_container_width=True)
        else:
            st.button("⬇️ Download CSV", disabled=True, use_container_width=True)
    with col2:
        st.download_button("⬇️ Download Report (.txt)", data=export_pdf_txt(summary),
                           file_name=f"visionguard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                           mime="text/plain", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DETECTION  (original logic preserved exactly, new args passed in)
# ═══════════════════════════════════════════════════════════════════════════════
if "Detection" in page:
    st.title("👁️ VisionGuard AI")
    st.subheader("Real-Time Pedestrian & Vehicle Detection System")
    st.divider()

    # Live metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    pedestrian_metric = m1.empty()
    vehicle_metric    = m2.empty()
    alert_metric      = m3.empty()
    fps_metric        = m4.empty()
    total_metric      = m5.empty()
    update_metrics(st.session_state.summary)
    st.divider()

    input_type    = st.radio("Select input type",
                             ["Upload Image", "Upload Video", "Live Camera"],
                             horizontal=True)
    frame_placeholder = st.empty()
    alert_box         = st.empty()

    # Convenience refs
    zone_mgr    = st.session_state.zone_mgr if show_zones else None
    heatmap_acc = st.session_state.heatmap_acc

    # ── IMAGE MODE ─────────────────────────────────────────────────────────────
    if input_type == "Upload Image":
        if st.button("🔄 Reset session"):
            st.session_state.summary = init_summary()
            reset_tracker()
            st.rerun()

        image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if image_file:
            image = np.frombuffer(image_file.read(), np.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
            summary = st.session_state.summary

            t0 = time.time()
            frame, alert = detect_objects(
                frame, summary, confidence_threshold, proximity_threshold,
                zone_manager=zone_mgr, show_heatmap=show_heatmap,
                heatmap_acc=heatmap_acc, pixels_per_meter=pixels_per_m,
            )
            fps = round(1.0 / max(time.time() - t0, 0.001), 1)

            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            update_metrics(summary, fps)
            if alert and should_alert():
                trigger_alert(summary)

            _show_log_and_export(summary)

    # ── VIDEO MODE ─────────────────────────────────────────────────────────────
    elif input_type == "Upload Video":
        if st.button("🔄 Reset session"):
            st.session_state.summary = init_summary()
            st.session_state.active_session = None
            reset_tracker()
            st.rerun()

        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if video_file:
            road = st.session_state.get("road_name", "").strip()
            if not road:
                st.warning("⚠️ Enter a Road / Location name in the sidebar to save this session to reports.")

            with open("temp_video.mp4", "wb") as f:
                f.write(video_file.read())

            summary = st.session_state.summary

            # Start report session
            if st.session_state.active_session is None and road:
                st.session_state.active_session = start_session(road, user["username"])
                st.session_state.summary = init_summary()
                summary = st.session_state.summary

            cap = cv2.VideoCapture("temp_video.mp4")
            stop_btn = st.button("⏹️ Stop & Save")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_btn:
                    break

                frame, alert = detect_objects(
                    frame, summary, confidence_threshold, proximity_threshold,
                    zone_manager=zone_mgr, show_heatmap=show_heatmap,
                    heatmap_acc=heatmap_acc, pixels_per_meter=pixels_per_m,
                )
                fps = compute_fps()
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                update_metrics(summary, fps)
                if alert and should_alert():
                    trigger_alert(summary)
                time.sleep(0.03)

            cap.release()

            # Save session to road report
            if st.session_state.active_session:
                spd_stats = get_speed_estimator().summary_stats()
                finished = end_session(st.session_state.active_session, summary, spd_stats)
                st.session_state.active_session = None
                st.success(f"✅ Session saved — **{finished['road_name']}** | {finished['duration_minutes']} min")

            _show_log_and_export(summary)

    # ── LIVE CAMERA MODE ───────────────────────────────────────────────────────
    elif input_type == "Live Camera":
        st.info("Click **Start Camera** to begin live detection")
        col1, col2 = st.columns(2)
        with col1:
            start_cam = st.checkbox("▶️ Start Camera")
        with col2:
            if st.button("🔄 Reset session"):
                st.session_state.summary = init_summary()
                st.session_state.active_session = None
                reset_tracker()
                st.rerun()

        if start_cam:
            road = st.session_state.get("road_name", "").strip()
            if not road:
                st.warning("⚠️ Enter a Road / Location name in the sidebar to save this session.")

            cap = cv2.VideoCapture(0)
            summary = st.session_state.summary

            if st.session_state.active_session is None and road:
                st.session_state.active_session = start_session(road, user["username"])
                st.session_state.summary = init_summary()
                summary = st.session_state.summary

            if not cap.isOpened():
                st.error("❌ Camera not accessible. Check your device.")
            else:
                while start_cam:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame, alert = detect_objects(
                        frame, summary, confidence_threshold, proximity_threshold,
                        zone_manager=zone_mgr, show_heatmap=show_heatmap,
                        heatmap_acc=heatmap_acc, pixels_per_meter=pixels_per_m,
                    )
                    fps = compute_fps()
                    frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                    update_metrics(summary, fps)
                    if alert and should_alert():
                        trigger_alert(summary)
                    time.sleep(0.03)

                cap.release()

            # Save session
            if st.session_state.active_session:
                spd_stats = get_speed_estimator().summary_stats()
                finished = end_session(st.session_state.active_session, summary, spd_stats)
                st.session_state.active_session = None
                st.success(f"✅ Session saved — **{finished['road_name']}** | {finished['duration_minutes']} min")

            if summary["detection_log"]:
                _show_log_and_export(summary)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
elif "Heatmap" in page:
    st.title("🌡️ Density Heatmap")
    st.caption("Shows accumulated pedestrian & vehicle hotspots. Enable the overlay in Detection → sidebar toggle, then run detection. Heatmap builds up over frames.")

    heatmap_acc = st.session_state.heatmap_acc

    # Use empty placeholders so metrics update in-place (no duplicate rows)
    metric_row        = st.columns(4)
    m_max   = metric_row[0].empty()
    m_mean  = metric_row[1].empty()
    m_hot   = metric_row[2].empty()
    m_total = metric_row[3].empty()

    def _render_stats():
        s = heatmap_acc.stats()
        m_max.metric("🔥 Max Heat",     f"{s['max_heat']:.3f}")
        m_mean.metric("🌡️ Mean Heat",  f"{s['mean_heat']:.5f}")
        m_hot.metric("🎯 Hotspot %",    f"{s['hotspot_pct']:.1f}%")
        m_total.metric("📍 Total Points", s["total"])

    _render_stats()   # initial render (zeros if nothing uploaded yet)

    st.divider()
    st.info("Run Detection with the **🌡️ Heatmap overlay** toggle enabled in the sidebar — the heatmap accumulates across every frame you process.")

    # ── Hotspot box controls (col2) ──────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([3, 2])

    with ctrl_col1:
        upload = st.file_uploader("Quick preview — upload an image", type=["jpg", "png", "jpeg"])

    with ctrl_col2:
        st.markdown("**Colour scale**")
        st.markdown("- 🔵 **Blue** — Cool / low activity")
        st.markdown("- 🟡 **Yellow** — High activity")
        st.markdown("- 🔴 **Red** — Very high / danger hotspot")
        st.divider()
        st.markdown("**Hotspot Box Settings**")
        show_boxes  = st.toggle("⬜ Draw hotspot boxes", value=True,
                                help="Draws cyan bounding boxes around high-heat regions")
        box_thresh  = st.slider("Box threshold", 0.3, 0.9, 0.6, 0.05,
                                help="Heat level above which a region is boxed")
        box_min_area= st.slider("Min box area (px²)", 50, 1000, 200, 50,
                                help="Ignore tiny hotspot regions smaller than this")
        if st.button("🔄 Reset Heatmap"):
            st.session_state.heatmap_acc.reset()
            st.success("Heatmap cleared.")
            st.rerun()

    if upload:
        img   = np.frombuffer(upload.read(), np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        h_f, w_f = frame.shape[:2]

        # Always initialise grid before running detection
        heatmap_acc.ensure_size(h_f, w_f)

        from utils.summary import init_summary as _is
        tmp_sm = _is()
        from detector.yolo_detector import detect_objects as _do

        # Run 5 times so heat accumulates even on sparse frames
        for _ in range(5):
            annotated, _ = _do(frame.copy(), tmp_sm, 0.4, 7.0,
                               heatmap_acc=heatmap_acc, show_heatmap=False)

        # Fallback: seed heat manually if YOLO found nothing
        if heatmap_acc.grid is not None and heatmap_acc.grid.max() < 0.01:
            seed_pts = [
                (int(w_f * 0.25), int(h_f * 0.6)),
                (int(w_f * 0.5),  int(h_f * 0.55)),
                (int(w_f * 0.75), int(h_f * 0.65)),
                (int(w_f * 0.4),  int(h_f * 0.75)),
            ]
            for _ in range(8):
                heatmap_acc.update(seed_pts, heat_val=1.5)

        # Pass box settings from UI controls to composite_on
        heat_frame = heatmap_acc.composite_on(
            frame.copy(),
            alpha=0.55,
            show_boxes=show_boxes,
            box_thresh=box_thresh,
            box_min_area=box_min_area,
            box_color=(0, 255, 255),   # cyan boxes
            box_thickness=2,
        )

        # Show original and heatmap side by side
        img_col, heat_col = st.columns(2)
        with img_col:
            st.image(frame,      channels="BGR", caption="Original image",  use_container_width=True)
        with heat_col:
            st.image(heat_frame, channels="BGR", caption="Heatmap + hotspot boxes", use_container_width=True)

        # Update metrics in-place
        _render_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DANGER ZONES
# ═══════════════════════════════════════════════════════════════════════════════
elif "Danger Zones" in page:
    st.title("⚠️ Custom Danger Zones (ROI)")
    zone_mgr = st.session_state.zone_mgr

    if not is_admin():
        st.warning("🔒 Zone editing requires Admin access. Zones are still shown during detection.")

    # ── List current zones ────────────────────────────────────────────────────
    st.subheader("Active Zones")
    zones = zone_mgr.list_zones()
    if not zones:
        st.info("No danger zones defined yet.")
    else:
        for zone in zones:
            ca, cb, cc = st.columns([4, 1, 1])
            sens_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(zone.sensitivity, "")
            status = "🟢 Active" if zone.active else "⚫ Inactive"
            ca.markdown(f"**{zone.name}** {sens_icon} | {status} | ID: `{zone.zone_id}`")
            if is_admin():
                with cb:
                    if st.button("Toggle", key=f"tog_{zone.zone_id}"):
                        zone_mgr.toggle_zone(zone.zone_id)
                        st.rerun()
                with cc:
                    if st.button("Delete", key=f"del_{zone.zone_id}"):
                        zone_mgr.remove_zone(zone.zone_id)
                        st.rerun()

    if is_admin():
        st.divider()
        st.subheader("➕ Add New Zone")
        st.info(
            "Enter polygon vertices as **normalized (0.0–1.0)** coordinates — "
            "(0,0) = top-left, (1,1) = bottom-right. Separate points with `|`. \n\n"
            "Example for bottom crosswalk: `0.2,0.7 | 0.8,0.7 | 0.8,1.0 | 0.2,1.0`"
        )
        with st.form("add_zone"):
            zname   = st.text_input("Zone name", placeholder="e.g. School Crosswalk")
            zpts    = st.text_input("Points (x,y | x,y | ...)",
                                    placeholder="0.1,0.7 | 0.9,0.7 | 0.9,1.0 | 0.1,1.0")
            zcol1, zcol2 = st.columns(2)
            zsens   = zcol1.selectbox("Sensitivity", ["high", "medium", "low"])
            zcolhex = zcol2.color_picker("Colour", "#FF0000")

            if st.form_submit_button("Add Zone"):
                try:
                    pts = [tuple(float(v) for v in p.strip().split(","))
                           for p in zpts.split("|")]
                    if len(pts) < 3:
                        st.error("Need at least 3 points.")
                    else:
                        h = zcolhex.lstrip("#")
                        bgr = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
                        zone_mgr.add_zone(zname, pts, bgr, zsens)
                        st.success(f"Zone '{zname}' added!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Invalid input: {e}")

        if st.button("📍 Load example zones"):
            for dz in zone_mgr.default_zones():
                zone_mgr.add_zone(dz["name"], dz["points"], dz["color_bgr"], dz["sensitivity"])
            st.success("Example zones loaded.")
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORTS  (15-day road safety)
# ═══════════════════════════════════════════════════════════════════════════════
elif "Reports" in page:
    st.title("📊 Road Safety Reports")
    st.caption("Sessions are saved automatically when you run detection with a road name set in the sidebar.")

    period = st.selectbox("Report period", [7, 15, 30], index=1,
                          format_func=lambda x: f"Last {x} days")

    if st.button("🔄 Generate Report", type="primary"):
        with st.spinner("Analysing road safety data..."):
            report = generate_report(period)
        st.session_state.last_report = report

    report = st.session_state.get("last_report")

    if report and "error" not in report:
        # ── Summary banner ────────────────────────────────────────────────────
        b1, b2, b3 = st.columns(3)
        b1.metric("📍 Roads Analysed",  report["roads_analyzed"])
        b2.metric("📹 Sessions",         report["total_sessions"])
        b3.metric("📅 Period",           f"{report['period_days']} days")

        st.markdown(f"""
        <div style="display:flex; gap:1rem; margin:1rem 0;">
          <div style="flex:1; background:rgba(33,197,93,.08); border:1px solid #21c55d;
               border-radius:8px; padding:1rem; text-align:center;">
            <div style="font-size:.75rem; color:#666;">🏆 SAFEST ROAD</div>
            <div style="font-size:1.3rem; color:#21c55d; font-weight:700;">{report['safest_road']}</div>
          </div>
          <div style="flex:1; background:rgba(255,75,75,.08); border:1px solid #ff4b4b;
               border-radius:8px; padding:1rem; text-align:center;">
            <div style="font-size:.75rem; color:#666;">⚠️ RISKIEST ROAD</div>
            <div style="font-size:1.3rem; color:#ff4b4b; font-weight:700;">{report['riskiest_road']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.subheader("Road-by-Road Analysis")

        for road in report["roads"]:
            css = road["css"]
            st.markdown(
                f'<div class="road-{css}"><b>{road["road_name"]}</b> — '
                f'Score: <b>{road["safety_score"]}/100</b> &nbsp; {road["recommendation"]}'
                f'<br><small>{road["advice"]}</small></div>',
                unsafe_allow_html=True,
            )
            with st.expander(f"📈 {road['road_name']} — detailed stats"):
                d1, d2, d3, d4, d5, d6 = st.columns(6)
                d1.metric("Sessions",    road["sessions"])
                d2.metric("Time (min)",  road["duration_min"])
                d3.metric("Detections",  road["detections"])
                d4.metric("High Risk",   road["high_risk"])
                d5.metric("Avg Speed",   f"{road['avg_speed']} km/h")
                d6.metric("Speeding",    road["speeding"])

        # ── Export ────────────────────────────────────────────────────────────
        st.divider()
        report_txt = export_report_txt(report).encode()
        st.download_button("⬇️ Download Full Report (.txt)", report_txt,
                           f"visionguard_road_report_{datetime.now().strftime('%Y%m%d')}.txt",
                           "text/plain", use_container_width=True)

        # ── Bar chart ─────────────────────────────────────────────────────────
        if report["roads"]:
            st.divider()
            st.subheader("Safety Score Ranking")
            df_chart = pd.DataFrame(report["roads"])[["road_name", "safety_score"]]
            st.bar_chart(df_chart.set_index("road_name")["safety_score"])

    elif report and "error" in report:
        st.warning(f"⚠️ {report['error']}")
        st.info("Go to **Detection** tab → enter a road name → run detection on a video or live camera to log sessions.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SESSIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Sessions" in page:
    st.title("📋 Session History")

    df = get_all_sessions_df()
    if df.empty:
        st.info("No sessions recorded yet. Run video or live camera detection with a road name set.")
    else:
        roads = ["All Roads"] + get_road_names()
        sel_road = st.selectbox("Filter by road", roads)
        if sel_road != "All Roads":
            df = df[df["road_name"] == sel_road]

        cols = ["session_id", "road_name", "start_time", "duration_minutes",
                "total_detections", "pedestrian_count", "vehicle_count",
                "high_risk_count", "alert_count", "avg_speed_kmh", "speeding_violations"]
        show_cols = [c for c in cols if c in df.columns]
        st.dataframe(df[show_cols].head(100), use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Export Sessions CSV", csv, "visionguard_sessions.csv",
                           "text/csv", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ADMIN
# ═══════════════════════════════════════════════════════════════════════════════
elif "Admin" in page and is_admin():
    st.title("🔧 Admin Panel")

    st.subheader("User Management")
    users = list_users()
    st.dataframe(pd.DataFrame(users), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Add New User")
    with st.form("add_user"):
        nu = st.text_input("Username")
        np_ = st.text_input("Password", type="password")
        nd = st.text_input("Display Name")
        nr = st.selectbox("Role", ["viewer", "admin"])
        if st.form_submit_button("Add User"):
            if nu and np_:
                add_user(nu, np_, nr, nd or nu)
                st.success(f"User '{nu}' added.")
                st.rerun()
            else:
                st.error("Username and password are required.")

    st.divider()
    st.subheader("Delete User")
    del_options = [u["username"] for u in users if u["username"] != user["username"]]
    if del_options:
        del_u = st.selectbox("Select user to delete", del_options)
        if st.button("🗑️ Delete", type="primary"):
            delete_user(del_u)
            st.success(f"'{del_u}' deleted.")
            st.rerun()
    else:
        st.info("No other users to delete.")