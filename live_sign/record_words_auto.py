import cv2
import mediapipe as mp
import numpy as np
import os
import time

SIGNS = [
    "HOW","YOU","THANK","YES","NO","PLEASE","SORRY",
    "HELP","GOOD","BAD","STOP","GO","COME","NAME"
]

SAMPLES_PER_SIGN = 20
FRAMES_PER_SAMPLE = 12

DATA_DIR = "live_sign/data_words"
os.makedirs(DATA_DIR, exist_ok=True)

VALUES_PER_FRAME = 126
EXPECTED_LEN = FRAMES_PER_SAMPLE * VALUES_PER_FRAME

REQUIRED_STABLE_FRAMES = 15

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

X, y = [], []
sign_index = 0
sample_count = 0
stable_frames = 0

print("\n=== AUTO WORD RECORDING (NO SPACE KEY) ===")
print("Show the sign → perform movement → hold ~2 seconds")
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
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    else:
        stable_frames = 0

    current_sign = SIGNS[sign_index]

    cv2.putText(frame, f"SIGN: {current_sign}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"SAMPLE: {sample_count}/{SAMPLES_PER_SIGN}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.imshow("Auto Record Words", frame)

    if stable_frames >= REQUIRED_STABLE_FRAMES:
        print(f"Auto-recording: {current_sign}")
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

        if len(sequence) < EXPECTED_LEN:
            sequence.extend([0.0] * (EXPECTED_LEN - len(sequence)))
        else:
            sequence = sequence[:EXPECTED_LEN]

        X.append(sequence)
        y.append(current_sign)
        sample_count += 1
        stable_frames = 0

        print(f"{current_sign} sample {sample_count} saved")
        time.sleep(0.8)  

    if sample_count == SAMPLES_PER_SIGN:
        sample_count = 0
        sign_index += 1

        if sign_index == len(SIGNS):
            break

        print(f"\nNEXT SIGN → {SIGNS[sign_index]}")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

np.save(os.path.join(DATA_DIR, "X.npy"), np.array(X, dtype=np.float32))
np.save(os.path.join(DATA_DIR, "y.npy"), np.array(y))

print("\n✅ WORD DATASET SAVED SUCCESSFULLY")
