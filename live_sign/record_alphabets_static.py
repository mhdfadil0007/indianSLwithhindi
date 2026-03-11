"""
record_alphabets_static.py

Purpose:
    Records static ASL alphabet hand signs (A-Z) using webcam and MediaPipe hand tracking.
    Auto-captures 50 samples per letter with enhanced feature extraction including:
    - Wrist-normalized 3D landmark coordinates (63 features)
    - Fingertip-to-base distances for each finger (5 features)
    - Inter-fingertip distances (10 features)
    - Joint angles at finger mid-points (4 features)
    
Total features per sample: 82 features

Usage:
    Run the script, show each letter to the webcam, hold steady - captures automatically.
    Press ESC to exit early.

Output:
    - X.npy: Feature array (N x 82)
    - y.npy: Label array (N,)
    Saved to live_sign/data_alphabets_static/
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

# List of ASL alphabet letters to capture
ALPHABETS = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
]

SAMPLES_PER_LETTER = 50     # Number of samples to capture per letter
DATA_DIR = "live_sign/data_alphabets_static"
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# MEDIAPIPE SETUP
# ============================================================================

# MediaPipe hands solution for hand landmark detection
# static_image_mode=True: Treats each frame as a static image (good for static poses)
# max_num_hands=2: Detect up to 2 hands
# min_detection_confidence=0.5: Minimum confidence for hand detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,   
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Initialize webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_enhanced_features(landmarks):
   
    features = []
    
    # ---- 1. Wrist-normalized 3D coordinates ----
    # Use wrist (landmark 0) as reference point to make features invariant to hand position
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    
    # For each of 21 landmarks, compute relative position from wrist
    for lm in landmarks:
        features.extend([lm.x - wrist[0], lm.y - wrist[1], lm.z - wrist[2]])
    
    # ---- 2. Fingertip-to-base distances ----
    # Indices of fingertips and their corresponding bases
    # Thumb: tip=4, base=2; Index: tip=8, base=5; Middle: tip=12, base=9; etc.
    fingertips = [4, 8, 12, 16, 20]
    finger_bases = [2, 5, 9, 13, 17]
    finger_mids = [6, 10, 14, 18]  # Middle joints
    
    # Calculate distance from each fingertip to its base (measures finger extension)
    for i in range(5):
        tip = np.array([landmarks[fingertips[i]].x, landmarks[fingertips[i]].y, landmarks[fingertips[i]].z])
        base = np.array([landmarks[finger_bases[i]].x, landmarks[finger_bases[i]].y, landmarks[finger_bases[i]].z])
        dist = np.linalg.norm(tip - base)
        features.append(dist)
    
    # ---- 3. Inter-fingertip distances ----
    # Calculate distances between all pairs of fingertips (10 pairs for 5 fingers)
    # This captures the relative positions of fingers
    for i in range(5):
        for j in range(i + 1, 5):
            p1 = np.array([landmarks[fingertips[i]].x, landmarks[fingertips[i]].y])
            p2 = np.array([landmarks[fingertips[j]].x, landmarks[fingertips[j]].y])
            dist = np.linalg.norm(p1 - p2)
            features.append(dist)
    
    # ---- 4. Joint angles ----
    # Calculate angle at each finger's middle joint (fingertip-mid-base)
    # This captures the bending angle of each finger
    for i, tip_idx in enumerate(fingertips):
        if i < len(finger_mids):
            tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y])
            mid = np.array([landmarks[finger_mids[i]].x, landmarks[finger_mids[i]].y])
            base = np.array([landmarks[finger_bases[i]].x, landmarks[finger_bases[i]].y])
            
            # Vectors from mid point to tip and base
            v1 = tip - mid
            v2 = base - mid
            
            # Calculate angle between vectors using dot product
            # np.clip ensures the value is in valid range for arccos
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1))
            features.append(angle)
    
    return features


# ============================================================================
# DATA RECORDING LOOP
# ============================================================================

# Initialize lists to store features and labels
X, y = [], []
letter_index = 0  # Current letter being recorded
sample_count = 0  # Samples collected for current letter

print("\n=== STATIC ALPHABET RECORDING (ENHANCED) ===")
print("Show letter → hold steady → auto capture")
print("ESC = Exit\n")

# Main recording loop - runs until all letters are captured or ESC is pressed
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for mirror-like view (more natural for user)
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB (MediaPipe expects RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe to detect hand landmarks
    results = hands.process(rgb)

    current_letter = ALPHABETS[letter_index]

    # ---- Draw hand landmarks on frame for visualization ----
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the frame
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # ---- Display current progress on screen ----
    cv2.putText(frame, f"LETTER: {current_letter}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"SAMPLE: {sample_count}/{SAMPLES_PER_LETTER}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    # Show the frame in a window
    cv2.imshow("Record Static Alphabets", frame)
    
    # Wait for key press (1ms delay, allows window to update)
    key = cv2.waitKey(1) & 0xFF

    # ---- Auto-capture when hand is detected ----
    # If hand landmarks are found, extract features and save
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        
        # Extract enhanced features from detected hand
        features = extract_enhanced_features(landmarks)
        
        # Save features and label if extraction was successful
        if len(features) > 0:
            X.append(features)
            y.append(current_letter)
            sample_count += 1
            print(f"{current_letter} sample {sample_count} saved (features: {len(features)})")
            
            # Small delay between captures to avoid duplicate frames
            time.sleep(0.08)

    # ---- Check if done with current letter ----
    # After collecting SAMPLES_PER_LETTER, move to next letter
    if sample_count == SAMPLES_PER_LETTER:
        sample_count = 0
        letter_index += 1

        # Check if we've captured all letters
        if letter_index == len(ALPHABETS):
            break

        print(f"\nNEXT LETTER → {ALPHABETS[letter_index]}")
        time.sleep(0.3)

    # ---- Exit on ESC key ----
    if key == 27:
        break


# ============================================================================
# SAVE RECORDED DATA
# ============================================================================

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Save features (X) and labels (y) as numpy arrays
# dtype=np.float32 for memory efficiency
np.save(os.path.join(DATA_DIR, "X.npy"), np.array(X, dtype=np.float32))
np.save(os.path.join(DATA_DIR, "y.npy"), np.array(y))

print(f"\n✅ STATIC ALPHABET DATASET SAVED: {len(X)} samples, {len(X[0]) if X else 0} features")
