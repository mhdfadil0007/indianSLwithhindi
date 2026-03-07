# 🤟 AI Sign Language Recognition & Translation Platform  
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

indianSLwithhindi/
│
├── manage.py
├── requirements.txt
├── db.sqlite3
│
├── A2SL/ → Main Django Project Folder
│ ├── init.py
│ ├── settings.py
│ ├── urls.py
│ ├── wsgi.py
│ ├── asgi.py
│ └── views.py
│
├── templates/ → HTML Files
│ ├── base.html
│ ├── index.html
│ └── animation.html
│
├── static/ → CSS / JS (if used)
│ ├── css/
│ └── js/
│
├── assets/ → Sign Language Videos
│ ├── A.mp4
│ ├── B.mp4
│ ├── Hello.mp4
│ ├── Thank.mp4
│ └── ...
│
├── media/ → (Optional uploads)
│
└── tf-env/ → Virtual Environment

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
- SVM (RBF kernel)
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

# 🎓 Developed As

Final Year AI-Based Communication System Project




# Changes i made
i changed the versions in requirements.txt to match my python version of 3.12.3
- changed in numpy,opencv,mediapipe,protobuf,tensorflow
- also added scikit-learn in requirements.txt
- i trained the model first by increasing the samples_per_letter to 50 from 30, reduced min_detection_confidence to 0.5 from 0.7, and reduced time.sleep from 0.4 to 0.2(in record_alphabet_static.py folder)
- but it was only showing number 25 in the detected sign instead of a value, even if new signs were shown(in live detection)
- this was because the value was not been converted to alphabet, and the number was straight introduced. and 25 was the letter 'z'
- even when i fixed it, the accuracy was off
- the model was broken, only 3.85 percent accuracy on training data itself
- so i retrained the model again, but the accuracy only pumped to 40 percent
- so i tried data augmentation(which means to add horizontally flipped versions of each sample) and also added probabiity=True for proper confidence scores(on train_alphabets_static.py)
- this increased the accuracy to 90 percent, but still some letters were not accurate
- so i added feature extraction to it and retrained the data again

# changes i made after the previous changes
- This is based for live detection model
- model was trained on asl, but the model needs to be trained in isl data. so changed the max_num_hands = 2 instead of 1 in record_alpabets_static.py
- the earlier model was trained on static alphabetic model(static_image_mode= true), but isl requires motion, so we change from training one frame to training 12 consecutive frames(sequential frames)
- for the model, we used flatten + SVM(similar to words)
- removed record_alphabets_static.py and train_alphabets_static.py and created new record and train sequence.py file(since the static was based on asl and was trained on static data)
- this time, i only focused on feature extraction first, and based on the accuracy, will check whether to add data augmentation or not
- also updates code in a2sl/ views.py
- i was met with an issue where the whole recording process was taking around 20 mins, which was very long
- this was because each frame(total of 50 frames for each letter) had a 15 frame stability to be processed, which was taking a lot of time(eg: 15 stability for 1 frame of letter 'a', till 50 frames )
- so i went with an option of reducing to 20 samples per letter and removed stability requirement(since letters now require motion), thereby reducing the time to 5-6 minutes for training

- i came back to the static data itself instead of sequential data, and the views.py(earlier updated for sequence mode) was reverted back to static mode for live detection
  


- now comes the word section, where i recorded and trained words in the records file with 68 percent accuracy
- but the signs were not picked up in ui, this was because there was a mismatch during training and in views
- training used raw landmark coordinated whereas in views, used wrist-normalized coordinates
- so changed views and made it similar to training method and reduced min detection and tracking confidence to 0.7 from 0.5(reduction part in mediapipe)
- still error was persistent, the output was shown but it was clearing fast without waiting for another hand signal
- this was due to aggressive buffer clearing, where after prediction the buffer is cleared immediately
- so inorder to fix that, sliding window logic is introduced and word_frame_buffer.clear()is removed(code in views.py)
- still the error is formed,so we changed views.py to event based prediction
- the whole process was completed, but the accuracy for predicting words was very low(and only worked on stable and not motion movements)
- so changed again in records and views.py to incorporate wrist normalization similar to alphabets
- again faced an error, recording didnt pick up. This was wrist = np.array([hand_landmarks[0]]) need to use .landmark[0] at the end
- now model accuracy increased from 64.5 to 92.86