import cv2
import mediapipe as mp
import numpy as np
import os

# MediaPipe Hand Landmarker setup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

def get_landmarker():
    mp_hands = mp.tasks.vision.HandLandmarker
    mp_base = mp.tasks.BaseOptions
    mp_vision = mp.tasks.vision

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_base(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1, 
        min_hand_detection_confidence=0.5
    )
    return mp_hands.create_from_options(options)

def analyze_palm(image_bgr):
    landmarker = get_landmarker()
    h, w, _ = image_bgr.shape
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return image_bgr, "Palm not detected", 0

    hand = result.hand_landmarks[0]
    
    # Landmark indices: 0=Wrist, 5=Index MCP, 17=Pinky MCP
    p0 = hand[0]  
    p5 = hand[5]  
    p17 = hand[17] 

    # Calculate center of the palm
    cx = int(((p0.x + p5.x + p17.x) / 3) * w)
    cy = int(((p0.y + p5.y + p17.y) / 3) * h)

    # Dynamic ROI size based on image height
    size = int(h * 0.08)
    x1, y1 = max(0, cx - size), max(0, cy - size)
    x2, y2 = min(w, cx + size), min(h, cy + size)
    
    palm_roi = image_bgr[y1:y2, x1:x2]

    if palm_roi.size == 0:
        return image_bgr, "ROI calculation error", 0

    # --- COLOR ANALYSIS ---
    # 1. LAB Space (a-channel focuses on Red/Green balance)
    lab = cv2.cvtColor(palm_roi, cv2.COLOR_BGR2LAB)
    a_channel = np.mean(lab[:, :, 1]) 
    
    # 2. Normalized Redness
    avg_bgr = np.mean(palm_roi, axis=(0, 1))
    b, g, r = avg_bgr
    norm_r = r / (r + g + b + 1e-6)

    # 3. Brightness
    hsv = cv2.cvtColor(palm_roi, cv2.COLOR_BGR2HSV)
    v = np.mean(hsv[:, :, 2]) / 255

    # Formula for Pallor Score (Higher = Paler)
    pallor_score = ((0.40 - norm_r) * 1.5 + (v * 0.2) + (1 - (a_channel / 200)) * 0.5)

    if pallor_score > 0.45:
        final_result = "High Anemia Risk (Severe Pallor)"
        color = (0, 0, 255) # Red BGR
        pale_status = 1
    elif pallor_score > 0.35:
        final_result = "Moderate Risk (Slight Pallor)"
        color = (0, 165, 255) # Orange BGR
        pale_status = 1
    else:
        final_result = "Normal (Healthy Pinkness)"
        color = (0, 255, 0) # Green BGR
        pale_status = 0

    # Draw result on frame
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 3)
    cv2.putText(image_bgr, f"Score: {round(pallor_score, 2)}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return image_bgr, final_result, pale_status