# Live Sign Language Detection Pipeline

## Overview

This document describes the complete pipeline for building a real-time sign language detection system using MediaPipe, scikit-learn, and Django.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RECORDING                                  │
│  record_alphabets_static.py                                      │
│  Camera → MediaPipe → 82 features → X.npy, y.npy               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING                                   │
│  train_alphabets_static.py                                      │
│  X.npy → Augment → Train SVC → alphabet_model.pkl             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      LIVE DETECTION                              │
│  Django server + templates/live_detect.html                      │
│  Camera → POST frame → Django → predict → JSON response         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Data Recording

### Script
`live_sign/record_alphabets_static.py`

### Purpose
Capture hand landmark data from webcam for each alphabet letter (A-Z).

### How It Works

1. **Camera Capture**: Opens webcam, displays live video
2. **Letter Display**: Shows which letter to record (A → Z sequentially)
3. **Auto-Capture**: Automatically captures samples at regular intervals
4. **Feature Extraction**: Uses MediaPipe to extract 21 hand landmarks

### MediaPipe Hand Landmarks

```
21 landmarks per hand, each with (x, y, z) coordinates = 63 raw features
```

| Landmark | Description |
|----------|-------------|
| 0 | Wrist |
| 1-4 | Thumb (CMC, IP, MCP, TIP) |
| 5-8 | Index Finger (MCP, PIP, DIP, TIP) |
| 9-12 | Middle Finger |
| 13-16 | Ring Finger |
| 17-20 | Pinky Finger |

### Enhanced Features (82 total)

| Feature Type | Count | Description |
|--------------|-------|-------------|
| Normalized Coordinates | 63 | MediaPipe (x, y, z) relative to wrist |
| Finger Lengths | 5 | Distance from fingertip to finger base |
| Fingertip Distances | 10 | Pairwise distances between fingertips |
| Finger Angles | 4 | Angles at finger joints |

### Running the Script

```bash
cd /home/adheeb/Desktop/projects/indianSLwithhindi
/home/adheeb/Desktop/projects/indianSLwithhindi/tf-env/bin/python live_sign/record_alphabets_static.py
```

### Output
- `live_sign/data_alphabets_static/X.npy` - Feature matrix (1300 × 82)
- `live_sign/data_alphabets_static/y.npy` - Labels (1300,)

### Configuration
- Samples per letter: 50
- Total letters: 26 (A-Z)
- Capture delay: 0.08 seconds between samples
- Letter change delay: 0.3 seconds

---

## Step 2: Model Training

### Script
`live_sign/train_alphabets_static.py`

### Purpose
Train an SVM classifier on the recorded data.

### How It Works

1. **Load Data**: Load X.npy and y.npy
2. **Augmentation**: Apply horizontal flip to double training data
3. **Encoding**: Convert letter labels to integers (A=0, B=1, ...)
4. **Split**: 80% training, 20% testing
5. **Pipeline**: StandardScaler → SVC (RBF kernel)
6. **Save**: Export model and label encoder

### Model Architecture

```python
Pipeline([
    ("scaler", StandardScaler()),      # Normalize features
    ("svm", SVC(                       # SVM classifier
        kernel="rbf",                   # Radial Basis Function
        C=10,                           # Regularization parameter
        gamma="scale",                  # Auto gamma
        probability=True                # Enable probability estimates
    ))
])
```

### Horizontal Flip Augmentation

```python
def augment_flip(X, y):
    X_flipped = X.copy()
    for i in range(len(X_flipped)):
        for j in range(21):
            idx = j * 3
            X_flipped[i, idx] = 1.0 - X_flipped[i, idx]      # Mirror x
            X_flipped[i, idx + 2] = -X_flipped[i, idx + 2]  # Invert z
    return np.vstack([X, X_flipped]), np.concatenate([y, y])
```

### Running the Script

```bash
/home/adheeb/Desktop/projects/indianSLwithhindi/tf-env/bin/python live_sign/train_alphabets_static.py
```

### Output

| File | Description |
|------|-------------|
| `live_sign/alphabet_model.pkl` | Trained SVM model |
| `live_sign/alphabet_labels.pkl` | Label encoder |

### Expected Results
- Samples: 1300 → 2600 (after augmentation)
- Features: 82
- Test Accuracy: ~95%

---

## Step 3: Live Detection (Django)

### Components

| File | Role |
|------|------|
| `A2SL/views.py` | Backend - processes frames, runs prediction |
| `templates/live_detect.html` | Frontend - camera, display, sentence formation |
| `A2SL/urls.py` | URL routing |

### Backend: views.py

```python
# 1. Receive frame from frontend
data = json.loads(request.body)
frame = data.get("frame", "")

# 2. Decode base64 image
header, encoded = frame.split(",", 1)
img_bytes = base64.b64decode(encoded)
np_arr = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# 3. Mirror flip (MUST match training)
img = cv2.flip(img, 1)

# 4. MediaPipe Hands (MUST match training config)
hands = mp_hands.Hands(
    static_image_mode=True,      # Same as training
    max_num_hands=1,
    min_detection_confidence=0.5
)

# 5. Extract enhanced features
features = extract_enhanced_features(landmarks)  # 82 features

# 6. Predict
pred = alphabet_model.predict(X)[0]
pred_letter = label_encoder.inverse_transform([int(pred)])[0]

# 7. Return JSON
return JsonResponse({
    "label": pred_letter,
    "confidence": float(confidence)
})
```

### Frontend: live_detect.html

**Features:**
- Webcam preview (getUserMedia API)
- Real-time frame capture (400ms interval)
- Sentence formation with pause detection
- History storage (localStorage)

**Sentence Formation Logic:**
- Letters accumulate as signs are detected
- 2-second pause → word break (space added)
- Confidence < 50% → ignored
- Same letter consecutively → skipped

**History Management:**
- Save sentences to browser localStorage
- Delete individual entries
- Persists across page refreshes

### Running the Server

```bash
cd /home/adheeb/Desktop/projects/indianSLwithhindi
python manage.py runserver
```

### Access
Open: `http://127.0.0.1:8000/live-detect/`

---

## Key Configuration Matching

| Parameter | Training | Inference (Django) |
|-----------|----------|-------------------|
| MediaPipe mode | `static_image_mode=True` | `static_image_mode=True` |
| Detection confidence | 0.5 | 0.5 |
| Image flip | `cv2.flip(frame, 1)` | `cv2.flip(img, 1)` |
| Features | 82 enhanced | 82 enhanced |

---

## Quick Commands Reference

| Step | Command |
|------|---------|
| Record data | `tf-env/bin/python live_sign/record_alphabets_static.py` |
| Train model | `tf-env/bin/python live_sign/train_alphabets_static.py` |
| Run server | `python manage.py runserver` |
| Test detection | `http://127.0.0.1:8000/live-detect/` |

---

## Troubleshooting

### Always predicting "Z"
- Check: Is `cv2.flip(img, 1)` in Django receive_frame?
- Check: Is `static_image_mode=True` in Django?

### Low accuracy
- Check: Feature count matches (82 in training vs inference)
- Check: MediaPipe confidence threshold
- Try: Re-record data with consistent hand positions

### Camera not working
- Check: Browser permissions for camera access
- Check: HTTPS requirement (some browsers require HTTPS for getUserMedia)

---

## Files Modified During Development

| File | Changes |
|------|---------|
| `A2SL/views.py` | Added flip, fixed MediaPipe config, enhanced features |
| `live_sign/record_alphabets_static.py` | Added enhanced feature extraction |
| `live_sign/train_alphabets_static.py` | Added augmentation, probability |
| `templates/live_detect.html` | Added sentence formation, history |
