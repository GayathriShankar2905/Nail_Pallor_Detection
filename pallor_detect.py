import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret: break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            
            # --- PALM CENTER LOGIC ---
            # Landmark 0: Wrist
            # Landmark 9: Middle finger MCP (base of the middle finger)
            wrist = hand_landmarks.landmark[0]
            mcp = hand_landmarks.landmark[9]

            # Calculate the center point of the palm
            palm_x = int((wrist.x + mcp.x) / 2 * w)
            palm_y = int((wrist.y + mcp.y) / 2 * h)

            # Define a larger ROI for the palm
            roi_size = 40 
            y1, y2 = max(0, palm_y - roi_size), min(h, palm_y + roi_size)
            x1, x2 = max(0, palm_x - roi_size), min(w, palm_x + roi_size)

            # Analyze the 'a' channel (Redness)
            roi_lab = image_lab[y1:y2, x1:x2]
            if roi_lab.size > 0:
                avg_redness = np.mean(roi_lab[:, :, 1])
                
                # Thresholding (Adjust 138 based on your lighting)
                status = "Normal" if avg_redness > 138 else "Possible Pallor"
                color = (0, 255, 0) if status == "Normal" else (0, 0, 255)

                # Visual Feedback
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (palm_x, palm_y), 5, (255, 255, 255), -1)
                
                cv2.putText(frame, f"Palm Redness: {round(avg_redness, 2)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Status: {status}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Optional: Draw the skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Palm Pallor Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()