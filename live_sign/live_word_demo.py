import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

model = joblib.load("live_sign/word_model.pkl")
encoder = joblib.load("live_sign/word_labels.pkl")

SIGNS = encoder.classes_

FRAMES_PER_SAMPLE = 12
VALUES_PER_FRAME = 126
EXPECTED_LEN = FRAMES_PER_SAMPLE * VALUES_PER_FRAME
REQUIRED_STABLE_FRAMES = 15

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

stable_frames = 0
prediction_text = "Show a word sign"

print("\n=== LIVE WORD SIGN RECOGNITION ===")
print("Perform word sign → hold hands steady → prediction appears")
print("ESC = Exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        stable_frames += 1
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        stable_frames = 0

    cv2.putText(frame, f"PREDICTION: {prediction_text}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Live Word Sign Recognition", frame)

    if stable_frames >= REQUIRED_STABLE_FRAMES:
        sequence = []

        for _ in range(FRAMES_PER_SAMPLE):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            frame_landmarks = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        frame_landmarks.extend([lm.x, lm.y, lm.z])

                if len(results.multi_hand_landmarks) == 1:
                    frame_landmarks.extend([0.0] * 63)
            else:
                frame_landmarks = [0.0] * VALUES_PER_FRAME

            sequence.extend(frame_landmarks)
            time.sleep(0.05)

        # Fix length
        if len(sequence) < EXPECTED_LEN:
            sequence.extend([0.0] * (EXPECTED_LEN - len(sequence)))
        else:
            sequence = sequence[:EXPECTED_LEN]

        sequence = np.array(sequence).reshape(1, -1)

        prediction = model.predict(sequence)[0]
        prediction_text = encoder.inverse_transform([prediction])[0]

        stable_frames = 0
        time.sleep(1)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
