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

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,   
    max_num_hands=1,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

def extract_enhanced_features(landmarks):
    features = []
    
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    
    for lm in landmarks:
        features.extend([lm.x - wrist[0], lm.y - wrist[1], lm.z - wrist[2]])
    
    fingertips = [4, 8, 12, 16, 20]
    finger_bases = [2, 5, 9, 13, 17]
    finger_mids = [6, 10, 14, 18]
    
    for i in range(5):
        tip = np.array([landmarks[fingertips[i]].x, landmarks[fingertips[i]].y, landmarks[fingertips[i]].z])
        base = np.array([landmarks[finger_bases[i]].x, landmarks[finger_bases[i]].y, landmarks[finger_bases[i]].z])
        dist = np.linalg.norm(tip - base)
        features.append(dist)
    
    for i in range(5):
        for j in range(i + 1, 5):
            p1 = np.array([landmarks[fingertips[i]].x, landmarks[fingertips[i]].y])
            p2 = np.array([landmarks[fingertips[j]].x, landmarks[fingertips[j]].y])
            dist = np.linalg.norm(p1 - p2)
            features.append(dist)
    
    for i, tip_idx in enumerate(fingertips):
        if i < len(finger_mids):
            tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y])
            mid = np.array([landmarks[finger_mids[i]].x, landmarks[finger_mids[i]].y])
            base = np.array([landmarks[finger_bases[i]].x, landmarks[finger_bases[i]].y])
            
            v1 = tip - mid
            v2 = base - mid
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1))
            features.append(angle)
    
    return features

X, y = [], []
letter_index = 0
sample_count = 0

print("\n=== STATIC ALPHABET RECORDING (ENHANCED) ===")
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
        landmarks = results.multi_hand_landmarks[0].landmark
        
        features = extract_enhanced_features(landmarks)
        
        if len(features) > 0:
            X.append(features)
            y.append(current_letter)
            sample_count += 1
            print(f"{current_letter} sample {sample_count} saved (features: {len(features)})")
            time.sleep(0.08)

    if sample_count == SAMPLES_PER_LETTER:
        sample_count = 0
        letter_index += 1

        if letter_index == len(ALPHABETS):
            break

        print(f"\nNEXT LETTER → {ALPHABETS[letter_index]}")
        time.sleep(0.3)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

np.save(os.path.join(DATA_DIR, "X.npy"), np.array(X, dtype=np.float32))
np.save(os.path.join(DATA_DIR, "y.npy"), np.array(y))

print(f"\n✅ STATIC ALPHABET DATASET SAVED: {len(X)} samples, {len(X[0]) if X else 0} features")
