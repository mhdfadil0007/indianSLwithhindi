import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

model = joblib.load("live_sign/alphabet_model.pkl")
encoder = joblib.load("live_sign/alphabet_labels.pkl")

VALUES_PER_FRAME = 63   
REQUIRED_STABLE_FRAMES = 15

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

stable_frames = 0
prediction_text = "Show an alphabet sign"

print("\n=== LIVE STATIC ALPHABET RECOGNITION ===")
print("Show letter → hold steady ~2 sec → prediction appears")
print("ESC = Exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    cv2.putText(frame, f"PREDICTION: {prediction_text}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    if results.multi_hand_landmarks:
        stable_frames += 1
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    else:
        stable_frames = 0

    cv2.imshow("Live Static Alphabet Recognition", frame)

    if stable_frames >= REQUIRED_STABLE_FRAMES:
        landmarks = []

        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == VALUES_PER_FRAME:
            sample = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(sample)[0]
            prediction_text = encoder.inverse_transform([prediction])[0]

        stable_frames = 0
        time.sleep(0.8)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
