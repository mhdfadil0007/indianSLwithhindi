import cv2
import mediapipe as mp
import numpy as np
import os
import time

ALPHABETS = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
]

SAMPLES_PER_LETTER = 50     
DATA_DIR = "live_sign/data_alphabets_static"
os.makedirs(DATA_DIR, exist_ok=True)

VALUES_PER_FRAME = 63       

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,   
    max_num_hands=1,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

X, y = [], []
letter_index = 0
sample_count = 0

print("\n=== STATIC ALPHABET RECORDING ===")
print("Show letter → hold steady → auto capture")
print("ESC = Exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_letter = ALPHABETS[letter_index]

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(frame, f"LETTER: {current_letter}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"SAMPLE: {sample_count}/{SAMPLES_PER_LETTER}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.imshow("Record Static Alphabets", frame)
    key = cv2.waitKey(1) & 0xFF

    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == VALUES_PER_FRAME:
            X.append(landmarks)
            y.append(current_letter)
            sample_count += 1
            print(f"{current_letter} sample {sample_count} saved")
            time.sleep(0.2)

    if sample_count == SAMPLES_PER_LETTER:
        sample_count = 0
        letter_index += 1

        if letter_index == len(ALPHABETS):
            break

        print(f"\nNEXT LETTER → {ALPHABETS[letter_index]}")
        time.sleep(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

np.save(os.path.join(DATA_DIR, "X.npy"), np.array(X, dtype=np.float32))
np.save(os.path.join(DATA_DIR, "y.npy"), np.array(y))

print("\n✅ STATIC ALPHABET DATASET SAVED SUCCESSFULLY")
