"""
=============================================================================
MODULE NAME: app.py
PROJECT: DriveMind AI // Cyberpunk Autonomous Co-Pilot & HUD Dashboard
DESCRIPTION:
    Streamlit-based professional cockpit interface featuring neon glassmorphism,
    real-time YOLO tracking, spatial sensor integration, AI brain telemetry,
    and stateful YouTube-style Pause/Resume simulation controls.
=============================================================================
"""

import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO
from sensor import VisionSensor
from brain import DriveBrain

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DriveMind AI // Autonomous Cockpit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CYBERPUNK NEON HUD STYLING ---
st.markdown("""
    <style>
    .stApp {
        background-color: #030712;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid #1e293b;
    }
    h1, h2, h3 {
        color: #00f3ff !important;
        font-weight: 900;
        letter-spacing: 1px;
        text-shadow: 0 0 20px rgba(0, 243, 255, 0.4);
    }
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 2px dashed #00f3ff;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 25px rgba(0, 243, 255, 0.15);
    }
    div[data-testid="stCheckbox"] {
        background: #0f172a;
        border: 1px solid #334155;
        padding: 12px 18px;
        border-radius: 12px;
    }
    div[data-testid="stExpander"] {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
    }
    .hud-telemetry-box {
        background: linear-gradient(145deg, #0b0f19 0%, #111827 100%);
        border-left: 5px solid #00f3ff;
        border-top: 1px solid #1f2937;
        border-right: 1px solid #1f2937;
        border-bottom: 1px solid #1f2937;
        padding: 16px;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        box-shadow: 0 8px 25px rgba(0,0,0,0.5);
        margin-bottom: 15px;
    }
    .stDownloadButton button {
        background: linear-gradient(90deg, #00f3ff 0%, #0066ff 100%) !important;
        color: #000000 !important;
        font-weight: 900 !important;
        border-radius: 10px !important;
        padding: 14px 28px !important;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ DRIVEMIND AI // AUTONOMOUS CO-PILOT")
st.markdown("### Multi-Layer Sensor & Decision Pipeline (YOLO → Vision Sensor → AI Brain)")

# --- SESSION STATE INITIALIZATION FOR STATEFUL PAUSE/RESUME ---
if "is_paused" not in st.session_state:
    st.session_state.is_paused = False
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "current_frame" not in st.session_state:
    st.session_state.current_frame = 0

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ COCKPIT CONTROLS")
uploaded_file = st.sidebar.file_uploader("📂 UPLOAD DASHCAM VIDEO (.MP4, .MOV)", type=["mp4", "avi", "mov"])
run = st.sidebar.checkbox("🚀 ENGAGE AI SIMULATION", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("⏯️ YOUTUBE-STYLE CONTROLS")
col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    if st.button("⏸️ STOP / PAUSE"):
        st.session_state.is_paused = True
        st.rerun()
with col_p2:
    if st.button("▶️ RESUME"):
        st.session_state.is_paused = False
        st.rerun()

with st.sidebar.expander("⚙️ SENSOR & HUD TUNING"):
    display_width = st.slider("Stream Resolution Width", 480, 1280, 720, step=50)
    skip_frames = st.selectbox("Performance Mode (Frame Skip)", [1, 2, 3], index=1)
st.sidebar.caption(
    "Vehicle labels: Bicycle, Car, Motorcycle (Bike), Bus, Truck. "
    "Auto-rickshaw detection requires a model trained for that class."
)

# Initialize Core Modules
@st.cache_resource
def load_modules():
    yolo_model = YOLO("yolov8n.pt")
    vision_sensor = VisionSensor()
    ai_brain = DriveBrain()
    return yolo_model, vision_sensor, ai_brain

model, sensor, brain = load_modules()

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
else:
    video_path = r"C:\Users\mohsi\Downloads\POV Dashcam Driving in City Traffic _ Realistic Urban Drive in India.mp4"

col1, col2 = st.columns([3, 1])

with col1:
    frame_placeholder = st.empty()

with col2:
    st.markdown("### 🧠 LIVE SENSOR TELEMETRY")
    status_box = st.empty()
    advice_box = st.empty()
    st.markdown("---")
    if st.session_state.is_paused:
        st.warning("⏸️ SYSTEM PAUSED. Video rok di gayi hai taaki aap advice parh saken.")
    else:
        st.info("🟢 Sensor array online. Strict spatial mapping active.")

if "trip_logs" not in st.session_state:
    st.session_state.trip_logs = []

# --- IF PAUSED, DISPLAY LAST CAPTURED FRAME AND STOP EXECUTION ---
if st.session_state.is_paused:
    if st.session_state.last_frame is not None:
        frame_placeholder.image(st.session_state.last_frame, channels="RGB", width="stretch")
    status_box.markdown("<div class='hud-telemetry-box'><b>STATUS:</b><br><span style='color: orange;'>SIMULATION PAUSED</span></div>", unsafe_allow_html=True)
    advice_box.markdown("<div class='hud-telemetry-box' style='border-left-color: orange;'><b>ADVICE:</b><br><span style='color: orange;'>Video ko rok diya gaya hai taaki aap aaram se telemetries aur advisories ko parh saken. Resume dabane par video wahin se agay chalegi jahan roki thi.</span></div>", unsafe_allow_html=True)
else:
    # --- MAIN SIMULATION LOOP ---
    if run:
        cap = cv2.VideoCapture(video_path)
        
        # Restore exact frame position if resumed from a paused state
        if st.session_state.current_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  
            
        frame_count = st.session_state.current_frame if st.session_state.current_frame > 0 else 0
        
        while cap.isOpened() and run:
            if st.session_state.is_paused:
                break

            success, frame = cap.read()
            if not success:
                st.warning("Video simulation stream completed.")
                break

            frame_count += 1
            st.session_state.current_frame = frame_count  # Keep tracking current frame position

            if frame_count % skip_frames != 0:
                continue

            timestamp_sec = round(frame_count / fps, 2)

            h, w, _ = frame.shape
            aspect_ratio = h / w
            new_width = display_width
            new_height = int(new_width * aspect_ratio)
            frame = cv2.resize(frame, (new_width, new_height))
            frame_height, frame_width, _ = frame.shape

            # 1. YOLO Detection & Tracking
            results = model.track(frame, persist=True, verbose=False)
            
            annotated_frame = frame.copy()
            boxes = None
            classes = None
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [None] * len(boxes)

                for box, cls, track_id in zip(boxes, classes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cls_id = int(cls)
                    box_width = x2 - x1
                    box_height = y2 - y1

                    if cls_id == 0:
                        label = "Person"
                        box_color = (255, 0, 128)
                    elif cls_id in [1, 2, 3, 5, 7]:
                        label = model.names[cls_id].title()
                        if cls_id == 3:
                            label = "Motorcycle (Bike)"
                        box_color = (0, 243, 255)
                    else:
                        continue

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    tid_str = f" #{int(track_id)}" if track_id is not None else ""
                    text = f"{label}{tid_str}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                    cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), box_color, -1)
                    cv2.putText(annotated_frame, text, (x1 + 5, y1 - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # 2. Vision Sensor Scanning
            sensor_report = sensor.scan_environment(boxes, classes, frame_width, frame_height)

            # 3. AI Brain Decision Making
            decision_status, decision_color, steering_advice = brain.make_decision(sensor_report)

            # Log entry
            st.session_state.trip_logs.append({
                "Time (s)": timestamp_sec,
                "Frame": frame_count,
                "System Status": decision_status,
                "Maneuver Advice": steering_advice,
                "Detected Objects": ", ".join(sensor_report["detected_objects"]) if sensor_report["detected_objects"] else "None"
            })

            status_box.markdown(f"<div class='hud-telemetry-box'><b>SENSOR STATUS:</b><br><span style='color: #00f3ff;'>{decision_status}</span></div>", unsafe_allow_html=True)
            advice_box.markdown(f"<div class='hud-telemetry-box' style='border-left-color: #0066ff;'><b>BRAIN ADVICE:</b><br><span style='color: #0066ff;'>{steering_advice}</span></div>", unsafe_allow_html=True)

            cv2.rectangle(annotated_frame, (0, 0), (frame_width, 65), (10, 15, 25), -1)
            cv2.putText(annotated_frame, f"SENSOR: {decision_status}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, decision_color, 2)
            cv2.putText(annotated_frame, f"ADVICE: {steering_advice}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.session_state.last_frame = annotated_frame_rgb
            frame_placeholder.image(annotated_frame_rgb, channels="RGB", width="stretch")

        cap.release()
    else:
        frame_placeholder.image("https://images.unsplash.com/photo-1506015391300-4802dc74de2e?q=80&w=1200&auto=format&fit=crop", 
                                caption="SYSTEM STANDBY. Upload dashcam video and engage simulation via sidebar controls.", 
                                width="stretch")

# --- TRIP TELEMETRY REPORT SECTION ---
if len(st.session_state.trip_logs) > 0:
    st.markdown("---")
    st.subheader("📊 SENSOR TELEMETRY & CHRONOLOGICAL TRIP REPORT")
    
    total_frames = len(st.session_state.trip_logs)
    traffic_events = sum(1 for log in st.session_state.trip_logs if "OBSTACLE" in log["System Status"] or "TRAFFIC" in log["System Status"] or "VEHICLE" in log["System Status"])
    evasive_suggestions = sum(1 for log in st.session_state.trip_logs if "SHIFT" in log["Maneuver Advice"] or "KHADI" in log["Maneuver Advice"] or "HONA" in log["Maneuver Advice"])
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Sensor Scans", total_frames)
    col_b.metric("Traffic & Obstacle Events", traffic_events)
    col_c.metric("Evasive Recommendations", evasive_suggestions)

    st.markdown("### 🕒 Chronological Sensor Log Table")
    st.dataframe(st.session_state.trip_logs, width="stretch", height=320)

    report_text = "=========================================================\n"
    report_text += "     DRIVEMIND AI - SENSOR TELEMETRY & TRIP REPORT       \n"
    report_text += "=========================================================\n\n"
    report_text += f"Total Scanned Frames: {total_frames}\n"
    report_text += f"Traffic/Obstacle Events: {traffic_events}\n"
    report_text += f"Evasive Recommendations: {evasive_suggestions}\n\n"
    report_text += "---------------------------------------------------------\n"
    report_text += "CHRONOLOGICAL SENSOR LOG:\n"
    report_text += "---------------------------------------------------------\n"
    
    for log in st.session_state.trip_logs:
        report_text += f"[Time: {log['Time (s)']}s | Frame: {log['Frame']}]\n"
        report_text += f"Status : {log['System Status']}\n"
        report_text += f"Advice : {log['Maneuver Advice']}\n"
        report_text += f"Objects: {log['Detected Objects']}\n"
        report_text += "---------------------------------------------------------\n"

    st.download_button(
        label="📥 DOWNLOAD CERTIFIED SENSOR TRIP REPORT (.TXT)",
        data=report_text, 
        file_name="DriveMind_AI_Sensor_Trip_Report.txt",
        mime="text/plain"
    )