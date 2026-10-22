import cv2
from ultralytics import YOLO

def main():
    # YOLOv8n model load karein
    model = YOLO("yolov8n.pt")
    
    # Aapka dashcam video path
    video_path = r"C:\Users\mohsi\Downloads\POV Dashcam Driving in City Traffic _ Realistic Urban Drive in India.mp4"
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video khatam ho gayi ya frame read nahi hua.")
            break

        frame_height, frame_width, _ = frame.shape

        # YOLOv8 tracking ke sath frame process karein
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
        annotated_frame = results[0].plot()

        # --- PROFESSIONAL DASHBOARD HUD OVERLAY (TOP BAR) ---
        # Upar ek dark strip banate hain jahan dashboard ki tarah info dikhe gi
        cv2.rectangle(annotated_frame, (0, 0), (frame_width, 60), (30, 30, 30), -1)
        cv2.putText(annotated_frame, "DriveMind AI Dashboard", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Status: MONITORING", (frame_width - 350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        warning_text = "SYSTEM NORMAL"
        warning_color = (0, 255, 0) # Green

        # Check karein ke objects detect hue hain ya nahi
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                if int(cls) in [2, 3, 5, 7]:  # Vehicles
                    x1, y1, x2, y2 = box
                    box_height = y2 - y1
                    x_center = (x1 + x2) / 2

                    # Center Lane check
                    is_in_center_lane = (frame_width * 0.35) < x_center < (frame_width * 0.65)

                    if is_in_center_lane and box_height > 220:
                        warning_text = "WARNING: APPLY BRAKE!"
                        warning_color = (0, 0, 255) # Red
                        break # Pehle khatray ko priority dein
                    elif not is_in_center_lane and box_height > 200:
                        if warning_text != "WARNING: APPLY BRAKE!":
                            warning_text = "CAUTION: SIDE VEHICLE"
                            warning_color = (0, 255, 255) # Yellow

        # Bottom ya side par live warning banner show karein
        cv2.rectangle(annotated_frame, (0, frame_height - 50), (frame_width, frame_height), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"ALERT: {warning_text}", (20, frame_height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, warning_color, 2)

        # OpenCV window mein smooth frame show karein
        cv2.imshow("DriveMind AI - Smart ADAS HUD", annotated_frame)

        # 'q' press karne par loop band ho jayega
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()