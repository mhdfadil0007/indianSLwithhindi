# AI Sign Language Recognition & Translation Platform  
### Full-Stack Web Application with Real-Time Gesture Detection

---

# 📌 1. Project Overview

The **AI Sign Language Recognition & Translation Platform** is a full-stack AI-powered web application designed to bridge communication between hearing-impaired individuals and non-sign users.

The system integrates:

- Natural Language Processing (NLP)
- Computer Vision (MediaPipe)
- Machine Learning (SVM Classifier)
- Real-Time Web Integration (Django + JavaScript)

The platform provides two major modules:

1️⃣ **Text → Sign Language Animation**  
2️⃣ **Live Sign Language → Text Detection (Webcam-Based)**  

---

# 🎯 2. Objectives

- Convert English and Hindi text into animated sign language
- Detect live hand gestures using webcam
- Classify static alphabets (A–Z)
- Recognize dynamic word gestures
- Integrate AI directly into a real-time web application
- Provide secure login-based access

---

# 🏗 3. Complete System Architecture

```
Frontend (HTML + CSS + JavaScript)
            ↓
Django Backend (Python)
            ↓
MediaPipe (Hand Landmark Detection)
            ↓
Feature Extraction (63 Features)
            ↓
SVM Machine Learning Model
            ↓
Real-Time Prediction Output
```



---

# 🔵 MODULE 1 — Text to Sign Animation

## 🔄 How It Works (Step-by-Step Flow)

```
User Input
     ↓
Django View (POST request)
     ↓
Language Check
     ↓
Translation (if Hindi)
     ↓
Tokenization (NLTK)
     ↓
Lemmatization
     ↓
Stopword Filtering
     ↓
Check if word.mp4 exists
     ↓
Send list of videos to frontend
     ↓
JavaScript plays animations sequentially
```

---

## 🌍 Translation Logic

If the user selects Hindi:

```python
translated_text = translator.translate(text, src="hi", dest="en").text.lower()
```

Symbolically:

```
Hindi Text
    ↓
Google Translator API
    ↓
English Text
```

Why translate to English?

Because animation videos are stored as:

```
how.mp4
are.mp4
you.mp4
```

So all input must be normalized into English for matching.

---

## 🧠 NLP Processing

Libraries Used:
- nltk
- googletrans
- gTTS

Processing includes:
- Tokenization
- POS Tagging
- Custom Lemmatization
- Stopword Filtering
- Protected verb preservation (e.g., "are" is not converted to "be")

Example:

Input:
```
How are you?
```

Processed:
```
["how", "are", "you"]
```

---

# 🟢 MODULE 2 — Live Sign Detection (AI Module)

## 🔄 Live Detection Flow

```
Webcam Frame
     ↓
JavaScript captures image
     ↓
Convert image → Base64
     ↓
Send to Django (/receive-frame/)
     ↓
Decode image (OpenCV)
     ↓
MediaPipe detects hand
     ↓
Extract 21 landmarks
     ↓
Convert to 63 numerical features
     ↓
SVM model predicts letter
     ↓
Return JSON response
     ↓
Browser updates UI
```

---

# 🧠 Feature Engineering

MediaPipe provides:

- 21 hand landmarks  
- Each landmark has (x, y, z)

So:

```
21 × 3 = 63 features
```

Instead of using raw image pixels (150,000+ values), we use 63 structured geometric features.

This makes the model:
- Faster
- Lightweight
- More efficient
- Easier to train

---

# 🤖 4. Machine Learning Implementation

## 🔹 Static Alphabet Model

- 63 landmark features
- SVM 
- StandardScaler preprocessing
- Label encoding
- 80/20 train-test split

### Why SVM?

- Works well for small datasets
- Good for structured numerical data
- Efficient for real-time prediction
- High accuracy for static gestures

---

## 🔹 Word Detection Model

- 12 frames per sample
- 126 features per frame (two hands)
- 1512 total features
- Multi-class SVM classifier

Why multiple frames?

Because words involve motion. A single frame cannot capture movement.

---

# 📊 5. Dataset Creation

Custom dataset recorders were built.

## Static Alphabet Dataset
- 20 samples per letter
- Total ≈ 780 samples
- Saved as X.npy and y.npy

## Word Dataset
- 20 samples per word

---


# 🛠 7. Technologies Used

## Backend
- Python
- Django
- OpenCV
- MediaPipe
- Scikit-learn
- NLTK
- Google Translate API
- gTTS

## Frontend
- HTML
- CSS
- JavaScript

## ML
- SVM (Support Vector Machine)
- StandardScaler
- LabelEncoder

---

# 📈 8. Accuracy

- Alphabet Model: ~90–95%
- Word Model: ~92%

Accuracy depends on:
- Lighting
- Hand stability
- Dataset quality
- Background noise

---

# ❓ Frequently Asked Questions (Doubt Clarification Section)

### Q1: Why not use Deep Learning (CNN)?

Deep learning requires:
- Large dataset
- GPU training
- More computation

Since we use structured landmark features (63 values), classical ML like SVM performs efficiently.

---

### Q2: Why not use raw images?

Raw images have thousands of features.

Landmark-based features:
- Reduce dimensionality
- Improve speed
- Improve generalization

---

### Q3: Why StandardScaler?

SVM is distance-based.
Feature scaling ensures all 63 features contribute equally.

---

### Q4: Why multiple frames for words?

Words involve motion.
Static frame cannot capture temporal patterns.

---

### Q5: What are the limitations?

- Sensitive to lighting
- May vary across users
- Not yet sentence-level dynamic detection
- Requires stable hand positioning

---

### Q6: How can this be improved?

- CNN + LSTM deep learning
- Larger dataset
- Multi-user training
- Majority voting for stability
- Cloud deployment

---

# 🔐 9. Authentication & Security

- Django login system
- Protected routes
- Session-based authentication

---

# ⚙ 10. Installation & Setup

## Step 1: Clone Repositor
```bash
git clone <repository-link>
cd project-folder
```

## Step 2: Create Virtual Environment

```bash
python -m venv tf-env
tf-env\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Run Server

```bash
python manage.py runserver
```

Access at:

```
http://127.0.0.1:8000
```

---

# 🎥 11. How to Use

## Text to Sign
1. Login
2. Go to Express
3. Enter sentence
4. Click submit
5. Watch animation

## Live Detection
1. Login
2. Go to Live Detect
3. Start camera
4. Show alphabet sign
5. View real-time prediction

---

# 📌 12. Project Status

- Text-to-Sign Module — Completed
- Live Alphabet Detection — Completed
- Word Detection — Integrated (Standalone + Training)
- UI Optimization — Completed
- Future Enhancements — Planned

---

# 🏆 Final Outcome

This project demonstrates:

- Full-stack AI integration
- Real-time computer vision
- NLP-based text processing
- Structured machine learning pipeline
- Modular architecture

It serves as a strong academic AI project and a scalable prototype for real-world assistive technology.

---

# 📎 License

Academic Project – Educational Use Only

---

# For better accuracy for words and alphabets
- Run the script in virtual environment that you create
```bash
cd live_sign
python record_alphabets_static.py
```
-  show each letter to the webcam, hold steady - captures automatically(make sure enough light is present when capturing letters)
-  After that , run this script in the same folder for training data
```bash
python train_alphabets_static.py
```
- for **words** part, simply exchange the file names for recording and training words in the script



