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

DATA_DIR = "live_sign/data_words_v2"
os.makedirs(DATA_DIR, exist_ok=True)

VALUES_PER_FRAME = 126
EXPECTED_LEN = FRAMES_PER_SAMPLE * VALUES_PER_FRAME

REQUIRED_DETECTION_FRAMES = 5

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
detection_frames = 0

print("\n=== AUTO WORD RECORDING (MOTION-BASED) ===")
print("Show the sign → perform movement → will capture automatically")
print("ESC = Exit\n")


def extract_wrist_normalized(landmarks_list):
    """Extract 126 wrist-normalized features from hand landmarks"""
    features = []
    
    if not landmarks_list:
        return [0.0] * VALUES_PER_FRAME
    
    for hand_landmarks in landmarks_list:
        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
        
        for lm in hand_landmarks.landmark:
            features.extend([lm.x - wrist[0], lm.y - wrist[1], lm.z - wrist[2]])
    
    if len(landmarks_list) == 1:
        features.extend([0.0] * 63)
    
    return features


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        detection_frames += 1
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    else:
        detection_frames = 0

    current_sign = SIGNS[sign_index]

    cv2.putText(frame, f"SIGN: {current_sign}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"SAMPLE: {sample_count}/{SAMPLES_PER_SIGN}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(frame, f"DETECTING: {detection_frames}/{REQUIRED_DETECTION_FRAMES}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("Auto Record Words", frame)

    if detection_frames >= REQUIRED_DETECTION_FRAMES:
        print(f"Recording: {current_sign}")
        sequence = []

        for _ in range(FRAMES_PER_SAMPLE):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            frame_landmarks = extract_wrist_normalized(results.multi_hand_landmarks)

            sequence.extend(frame_landmarks)
            time.sleep(0.05)

        if len(sequence) < EXPECTED_LEN:
            sequence.extend([0.0] * (EXPECTED_LEN - len(sequence)))
        else:
            sequence = sequence[:EXPECTED_LEN]

        X.append(sequence)
        y.append(current_sign)
        sample_count += 1
        detection_frames = 0

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
