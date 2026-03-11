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
5. **Pipeline**: StandardScaler → SVM
6. **Save**: Export model and label encoder


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

### Running the Server

```bash
cd /home/adheeb/Desktop/projects/indianSLwithhindi
python manage.py runserver
```

### Access
Open: `http://127.0.0.1:8000/live-detect/`

---



## Quick Commands Reference

| Step | Command |
|------|---------|
| Record data | `tf-env/bin/python live_sign/record_alphabets_static.py` |
| Train model | `tf-env/bin/python live_sign/train_alphabets_static.py` |
| Run server | `python manage.py runserver` |
| Test detection | `http://127.0.0.1:8000/live-detect/` |

---


