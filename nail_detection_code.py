import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe hand detector
mp_hands = mp.tasks.vision.HandLandmarker
mp_base = mp.tasks.BaseOptions
mp_vision = mp.tasks.vision
mp_running = mp.tasks.vision.RunningMode

# Load the hand landmark model
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_base(model_asset_path="hand_landmarker.task"),
    running_mode=mp_running.VIDEO,
    num_hands=1
)

landmarker = mp_hands.create_from_options(options)

cap = cv2.VideoCapture(0)

frame_timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect_for_video(mp_image, frame_timestamp)

    frame_timestamp += 1

    if result.hand_landmarks:

        h, w, _ = frame.shape

        for hand in result.hand_landmarks:

            fingertip_ids = [4, 8, 12, 16, 20]

            for tip in fingertip_ids:

                x = int(hand[tip].x * w)
                y = int(hand[tip].y * h)

                cv2.circle(frame, (x, y), 6, (0,255,0), -1)

                size = 20

                cv2.rectangle(
                    frame,
                    (x-size, y-size),
                    (x+size, y+size),
                    (255,0,0),
                    2
                )

                roi = frame[y-size:y+size, x-size:x+size]

                if roi.size > 0:
                    avg = np.mean(roi, axis=(0,1))
                    b,g,r = avg

                    text = f"R:{int(r)}"
                    cv2.putText(frame, text, (x-size, y-size-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imshow("Hand + Nail Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()