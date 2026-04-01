import cv2
import mediapipe as mp
import numpy as np

# =========================
# MediaPipe Setup (LOAD ONCE)
# =========================
mp_hands = mp.tasks.vision.HandLandmarker
mp_base = mp.tasks.BaseOptions
mp_vision = mp.tasks.vision

options = mp_vision.HandLandmarkerOptions(
    base_options=mp_base(model_asset_path="hand_landmarker.task"),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3
)

# GLOBAL LANDMARKER (IMPORTANT)
landmarker = mp_hands.create_from_options(options)

# Fingertip landmark IDs
fingertip_ids = [4, 8, 12, 16, 20]


# =========================
# MAIN FUNCTION
# =========================
def analyze_nails(image_bgr):

    h, w, _ = image_bgr.shape
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    pale_count = 0
    total_nails = 0

    # If no hands detected
    if not result.hand_landmarks:
        return image_bgr, "Hand not detected properly", 0

    for hand in result.hand_landmarks:
        for idx in fingertip_ids:

            lm = hand[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)

            # DEBUG POINT (shows fingertip detection)
            cv2.circle(image_bgr, (x, y), 5, (255, 0, 0), -1)

            # =========================
            # ROI (Improved box)
            # =========================
            size = int(min(w, h) * 0.025)  # adaptive size

            x1, y1 = max(0, x - size), max(0, y - size)
            x2, y2 = min(w, x + size), min(h, y + size)

            nail_roi = image_bgr[y1:y2, x1:x2]

            if nail_roi.size == 0:
                continue

            total_nails += 1

            # =========================
            # COLOR ANALYSIS (Improved)
            # =========================
            avg_bgr = np.mean(nail_roi, axis=(0, 1))
            b, g, r = avg_bgr

            # Normalize Red
            norm_r = r / (r + g + b + 1e-6)

            # HSV Brightness
            hsv = cv2.cvtColor(nail_roi, cv2.COLOR_BGR2HSV)
            v = np.mean(hsv[:, :, 2]) / 255

            # LAB color (more accurate for pallor)
            lab = cv2.cvtColor(nail_roi, cv2.COLOR_BGR2LAB)
            a_channel = np.mean(lab[:, :, 1])  # redness indicator

            # =========================
            # FINAL PALLOR SCORE (Improved formula)
            # =========================
            pallor_score = (
                (1 - norm_r) * 0.5 +
                (v) * 0.2 +
                (1 - (a_channel / 255)) * 0.3
            )

            # =========================
            # CLASSIFICATION
            # =========================
            if pallor_score > 0.65:
                risk = "High"
                color = (0, 0, 255)
                pale_count += 1
            elif pallor_score > 0.5:
                risk = "Moderate"
                color = (0, 165, 255)
            else:
                risk = "Normal"
                color = (0, 255, 0)

            # =========================
            # DRAW BOX + LABEL
            # =========================
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image_bgr,
                risk,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

    # =========================
    # FINAL DECISION
    # =========================
    if total_nails == 0:
        return image_bgr, "Fingertips not clearly visible", 0

    if pale_count >= 3:
        final = "High Anemia Risk"
    elif pale_count == 2:
        final = "Moderate Risk"
    else:
        final = "Normal"

    return image_bgr, final, pale_count