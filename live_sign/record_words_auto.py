"""
record_words_auto.py

Purpose:
    Records dynamic word signs (multi-frame gestures) using webcam and MediaPipe.
    Uses motion-based detection to automatically capture samples when hand is detected
    and stable for a few frames.
    
    Each sample consists of 12 consecutive frames, each with 126 features:
    - 63 features for first hand (wrist-normalized 3D coordinates)
    - 63 features for second hand (or zeros if only one hand)
    
Total features per sample: 12 frames × 126 features = 1512 features

Words captured:
    HOW, YOU, THANK, YES, NO, PLEASE, SORRY, HELP, GOOD, BAD, STOP, GO, COME, NAME

Usage:
    Run the script, show the sign gesture, perform the movement.
    The system auto-captures when hand is detected steadily for 5 frames.
    Press ESC to exit early.

Output:
    - X.npy: Feature array (N x 1512)
    - y.npy: Label array (N,)
    Saved to live_sign/data_words_v2/
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

# List of word signs to capture
SIGNS = [
    "HOW","YOU","THANK","YES","NO","PLEASE","SORRY",
    "HELP","GOOD","BAD","STOP","GO","COME","NAME"
]

SAMPLES_PER_SIGN = 20      # Number of gesture samples per word
FRAMES_PER_SAMPLE = 12    # Number of frames to capture per gesture sample

DATA_DIR = "live_sign/data_words_v2"
os.makedirs(DATA_DIR, exist_ok=True)

# ---- Feature configuration ----
# 21 landmarks × 3 coordinates (x, y, z) = 63 values per hand
VALUES_PER_FRAME = 126   # 63 for first hand + 63 for second hand
EXPECTED_LEN = FRAMES_PER_SAMPLE * VALUES_PER_FRAME  # 12 × 126 = 1512

# ---- Motion detection threshold ----
# Number of consecutive frames with hand detection required before auto-capture
REQUIRED_DETECTION_FRAMES = 5


# ============================================================================
# MEDIAPIPE SETUP
# ============================================================================

# MediaPipe hands solution
# static_image_mode=False: Tracks hands across frames (better for motion detection)
# min_tracking_confidence=0.7: Minimum confidence for landmark tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize webcam
cap = cv2.VideoCapture(0)


# ============================================================================
# DATA RECORDING STATE
# ============================================================================

X, y = [], []
sign_index = 0       # Current word being recorded
sample_count = 0    # Samples collected for current word
detection_frames = 0  # Consecutive frames with hand detected

print("\n=== AUTO WORD RECORDING (MOTION-BASED) ===")
print("Show the sign → perform movement → will capture automatically")
print("ESC = Exit\n")


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_wrist_normalized(landmarks_list):
   
    features = []
    
    # If no hands detected, return zeros
    if not landmarks_list:
        return [0.0] * VALUES_PER_FRAME
    
    # Process each detected hand
    for hand_landmarks in landmarks_list:
        # Get wrist position as reference point
        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
        
        # Compute wrist-normalized coordinates for all 21 landmarks
        for lm in hand_landmarks.landmark:
            features.extend([lm.x - wrist[0], lm.y - wrist[1], lm.z - wrist[2]])
    
    # If only one hand, pad with zeros for second hand
    if len(landmarks_list) == 1:
        features.extend([0.0] * 63)
    
    return features


# ============================================================================
# MAIN RECORDING LOOP
# ============================================================================

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hand landmarks
    results = hands.process(rgb)

    # ---- Track consecutive detection frames ----
    # This ensures hand is stable before recording
    if results.multi_hand_landmarks:
        detection_frames += 1
        # Draw hand landmarks for visualization
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    else:
        # Reset counter if hand is lost
        detection_frames = 0

    current_sign = SIGNS[sign_index]

    # Display progress on screen
    cv2.putText(frame, f"SIGN: {current_sign}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"SAMPLE: {sample_count}/{SAMPLES_PER_SIGN}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(frame, f"DETECTING: {detection_frames}/{REQUIRED_DETECTION_FRAMES}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # Show video feed
    cv2.imshow("Auto Record Words", frame)

    # ---- Auto-capture when hand detected steadily ----
    # Once hand is detected for REQUIRED_DETECTION_FRAMES consecutive frames,
    # start capturing the gesture sequence
    if detection_frames >= REQUIRED_DETECTION_FRAMES:
        print(f"Recording: {current_sign}")
        sequence = []

        # Capture FRAMES_PER_SAMPLE consecutive frames
        for _ in range(FRAMES_PER_SAMPLE):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Extract features from current frame
            frame_landmarks = extract_wrist_normalized(results.multi_hand_landmarks)

            # Add frame features to sequence
            sequence.extend(frame_landmarks)
            time.sleep(0.05)  # Small delay between frames

        # ---- Ensure sequence has expected length ----
        # Pad with zeros if less than expected, truncate if more
        if len(sequence) < EXPECTED_LEN:
            sequence.extend([0.0] * (EXPECTED_LEN - len(sequence)))
        else:
            sequence = sequence[:EXPECTED_LEN]

        # Save the complete gesture sequence
        X.append(sequence)
        y.append(current_sign)
        sample_count += 1
        detection_frames = 0  # Reset for next sample

        print(f"{current_sign} sample {sample_count} saved")
        time.sleep(0.8)  # Delay before next capture

    # ---- Check if done with current word ----
    if sample_count == SAMPLES_PER_SIGN:
        sample_count = 0
        sign_index += 1

        # Exit if all words captured
        if sign_index == len(SIGNS):
            break

        print(f"\nNEXT SIGN → {SIGNS[sign_index]}")

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ============================================================================
# SAVE RECORDED DATA
# ============================================================================

cap.release()
cv2.destroyAllWindows()

# Save features and labels as numpy arrays
np.save(os.path.join(DATA_DIR, "X.npy"), np.array(X, dtype=np.float32))
np.save(os.path.join(DATA_DIR, "y.npy"), np.array(y))

print("\n✅ WORD DATASET SAVED SUCCESSFULLY")
