
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import time
from ultralytics import YOLO

st.set_page_config(
    page_title="DriveMind AI",
    page_icon="🚘",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #edf2ff;
}
.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
}
.dm-header {
    padding: 24px;
    border: 1px solid #263451;
    border-radius: 16px;
    background: linear-gradient(120deg, #121e36, #101522);
    margin-bottom: 20px;
}
.dm-title {
    font-size: 36px;
    font-weight: 800;
    color: #f4f7ff;
}
.dm-subtitle {
    color: #aab8d2;
    font-size: 15px;
}
.metric-card {
    background: #121a2d;
    border: 1px solid #263451;
    border-radius: 12px;
    padding: 17px;
    min-height: 100px;
}
.metric-label {
    color: #aab8d2;
    font-size: 13px;
}
.metric-value {
    color: #ffffff;
    font-size: 26px;
    font-weight: 750;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dm-header">
    <div class="dm-title">🚘 DriveMind AI</div>
    <div class="dm-subtitle">
        Dashcam Intelligence • Object Detection • Traffic Hazard Analysis
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None

if "results" not in st.session_state:
    st.session_state.results = None

if "report" not in st.session_state:
    st.session_state.report = None

st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)

process_every = st.sidebar.select_slider(
    "Processing Speed",
    options=[1, 2, 3, 5],
    value=2,
    format_func=lambda x: f"Every {x} frame(s)"
)

show_labels = st.sidebar.checkbox("Show Object Labels", value=True)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)

st.sidebar.caption(
    "First run may download the pretrained YOLO model. "
    "Processing speed depends on your computer."
)

uploaded_video = st.file_uploader(
    "📁 Upload a dashcam video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is None:
    st.info(
        "Upload a dashcam video to begin. "
        "A short clip is recommended for your first test."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">VISION ENGINE</div>
            <div class="metric-value">YOLO</div>
            Pretrained object detection
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">VIDEO ANALYSIS</div>
            <div class="metric-value">OpenCV</div>
            Frame-by-frame processing
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">SAFETY MODULE</div>
            <div class="metric-value">Hazard AI</div>
            Preliminary visual risk alerts
        </div>
        """, unsafe_allow_html=True)

    st.stop()

st.video(uploaded_video)

if st.button("🚀 START AI ANALYSIS", type="primary", use_container_width=True):
    temp_path = None

    try:
        # Save uploaded video temporarily
        suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_file.write(uploaded_video.getvalue())
            temp_path = temp_file.name

        progress = st.progress(0)
        status = st.empty()

        video_col, info_col = st.columns([2, 1])

        with video_col:
            st.subheader("AI Vision Feed")
            frame_placeholder = st.empty()

        with info_col:
            st.subheader("Live Analysis")
            object_placeholder = st.empty()
            hazard_placeholder = st.empty()

        # Load pretrained model
        if st.session_state.model is None:
            status.info("Loading pretrained YOLO model...")
            st.session_state.model = YOLO("yolo11n.pt")

        model = st.session_state.model

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            st.error("Video open nahi ho saka. Doosri MP4/AVI file try karo.")
            st.stop()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            fps = 25.0

        if total_frames <= 0:
            total_frames = 1

        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.caption(
            f"Video: {video_width} × {video_height} | "
            f"FPS: {fps:.1f} | Frames: {total_frames}"
        )

        frame_index = 0
        processed_frames = 0
        detected_objects = []
        report_rows = []
        start_time = time.time()
        max_frames = 600

        while cap.isOpened() and processed_frames < max_frames:
            success, frame = cap.read()

            if not success:
                break

            frame_index += 1

            if (frame_index - 1) % process_every != 0:
                continue

            result = model.track(
                frame,
                persist=True,
                conf=confidence,
                tracker="bytetrack.yaml",
                verbose=False
            )[0]

            annotated = frame.copy()
            names = model.names
            frame_objects = []
            closest_vehicle_risk = 0.0

            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    label = names.get(class_id, str(class_id))
                    score = float(box.conf[0].item())

                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0].item())

                    object_record = {
                        "Frame": frame_index,
                        "Time (s)": round(frame_index / fps, 2),
                        "Object": label,
                        "Confidence": round(score, 3),
                        "Track ID": track_id if track_id is not None else "-"
                    }

                    detected_objects.append(object_record)
                    frame_objects.append(object_record)

                    # Bounding-box visibility controls
                    if show_boxes:
                        color = (60, 220, 130)

                        if label in ["car", "truck", "bus", "motorcycle"]:
                            color = (30, 180, 255)

                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )

                    if show_labels:
                        label_text = f"{label} {score:.0%}"
                        if track_id is not None:
                            label_text += f" ID:{track_id}"

                        cv2.putText(
                            annotated,
                            label_text,
                            (max(5, x1), max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2
                        )

                    # Approximate visual risk indicator.
                    # This is NOT a real distance or collision prediction.
                    vehicle_classes = ["car", "truck", "bus", "motorcycle"]
                    if label in vehicle_classes:
                        box_area = max(0, x2 - x1) * max(0, y2 - y1)
                        frame_area = max(1, video_width * video_height)
                        area_ratio = box_area / frame_area

                        box_center_x = (x1 + x2) / 2
                        box_center_y = (y1 + y2) / 2

                        central_zone = (
                            video_width * 0.25 <= box_center_x
                            <= video_width * 0.75
                        )

                        lower_zone = box_center_y > video_height * 0.40

                        if central_zone and lower_zone:
                            risk = min(100, area_ratio * 500)
                            closest_vehicle_risk = max(
                                closest_vehicle_risk, risk
                            )

            # Visual status overlay
            cv2.putText(
                annotated,
                "DRIVEMIND AI | VISUAL ANALYSIS",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(
                rgb_frame,
                channels="RGB",
                use_container_width=True
            )

            unique_objects = sorted(
                set(item["Object"] for item in frame_objects)
            )

            object_placeholder.write({
                "Frame": frame_index,
                "Objects detected": len(frame_objects),
                "Object types": ", ".join(unique_objects)
                if unique_objects else "None"
            })

            if closest_vehicle_risk >= 15:
                hazard_placeholder.warning(
                    "VISUAL ALERT: A vehicle occupies a relatively large "
                    "part of the central/lower image. Check manually."
                )
                hazard_status = "Visual caution"
            else:
                hazard_placeholder.success(
                    "No high visual-size alert in this processed frame."
                )
                hazard_status = "No size alert"

            report_rows.append({
                "Frame": frame_index,
                "Time (s)": round(frame_index / fps, 2),
                "Detected objects": len(frame_objects),
                "Object types": ", ".join(unique_objects),
                "Visual alert": hazard_status
            })

            processed_frames += 1

            progress.progress(
                min(1.0, frame_index / total_frames)
            )
            status.info(
                f"Processing frame {frame_index}/{total_frames} "
                f"• Analyzed {processed_frames} frames"
            )

        cap.release()

        if processed_frames >= max_frames:
            st.warning(
                "First test limit reached: 600 frames maximum. "
                "Use a shorter clip or increase the limit in app.py "
                "after testing."
            )

        elapsed = time.time() - start_time

        st.session_state.results = pd.DataFrame(detected_objects)
        st.session_state.report = pd.DataFrame(report_rows)

        status.success(
            f"Analysis complete: {processed_frames} frames analyzed "
            f"in {elapsed:.1f} seconds."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Frames Analyzed", processed_frames)
        col2.metric("Object Detections", len(detected_objects))
        col3.metric(
            "Unique Object Types",
            len(set(x["Object"] for x in detected_objects))
        )

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

if st.session_state.results is not None:
    st.divider()
    st.subheader("📊 Detection Results")

    results_df = st.session_state.results

    if not results_df.empty:
        counts = (
            results_df["Object"]
            .value_counts()
            .rename_axis("Object")
            .reset_index(name="Detections")
        )

        st.dataframe(counts, use_container_width=True, hide_index=True)

        with st.expander("View all object detections"):
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            "⬇️ Download Detection CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="drivemind_detections.csv",
            mime="text/csv"
        )
    else:
        st.warning("Is video mein selected confidence par objects detect nahi huay.")

if st.session_state.report is not None:
    st.subheader("🛣️ Frame-by-Frame Visual Report")

    report_df = st.session_state.report

    if not report_df.empty:
        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Analysis Report",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="drivemind_visual_report.csv",
            mime="text/csv"
        )

st.divider()
st.caption(
    "DriveMind AI is a research and decision-support prototype. "
    "Visual-size alerts are approximate, not calibrated distance or "
    "time-to-collision estimates. Do not use these outputs to control "
    "a real vehicle or replace a driver's judgment."
)
