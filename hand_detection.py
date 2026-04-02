import cv2
import mediapipe as mp
import numpy as np

# Setup MediaPipe Hand Landmarker
mp_hands = mp.tasks.vision.HandLandmarker
mp_base = mp.tasks.BaseOptions
mp_vision = mp.tasks.vision
mp_running = mp.tasks.vision.RunningMode

options = mp_vision.HandLandmarkerOptions(
    base_options=mp_base(model_asset_path="hand_landmarker.task"),
    running_mode=mp_running.VIDEO,
    num_hands=1
)

landmarker = mp_hands.create_from_options(options)
cap = cv2.VideoCapture(0)
frame_timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Detect landmarks
    result = landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            xs = []
            ys = []

            # Extract coordinates for landmarks (1-20, ignoring wrist)
            for i in range(1, 21):
                # MediaPipe returns normalized coordinates (0.0 to 1.0)
                x = int(hand[i].x * w)
                y = int(hand[i].y * h)
                xs.append(x)
                ys.append(y)

            # Define Bounding Box with Padding
            padding = 20
            x_min, x_max = max(0, min(xs) - padding), min(w, max(xs) + padding)
            y_min, y_max = max(0, min(ys) - padding), min(h, max(ys) + padding)

            # Draw rectangle on main frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Extract ROI
            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size > 0:
                # --- PALLOR ANALYSIS LOGIC (Must stay inside the 'if' block) ---
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                avg_bgr = np.mean(roi, axis=(0, 1))
                b, g, r = avg_bgr
                
                # Calculate normalized red channel
                norm_r = r / (r + g + b + 1e-6) 
                
                # Simplified Pallor Metric
                pallor = 1 - norm_r

                if pallor > 0.6:
                    risk = "High Anemia Risk"
                elif pallor > 0.45:
                    risk = "Moderate Risk"
                else:
                    risk = "Low Risk"

                # Display Stats
                text_bgr = f"R:{int(r)} G:{int(g)} B:{int(b)}"
                cv2.putText(frame, text_bgr, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.putText(frame, f"Pallor: {pallor:.2f}", (x_min, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.putText(frame, risk, (x_min, y_min - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Palm + Fingers ROI", roi)

    cv2.imshow("Anemia Risk Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27: # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()