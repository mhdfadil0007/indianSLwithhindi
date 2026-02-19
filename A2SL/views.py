import os
import json
import base64
import joblib
import numpy as np
import cv2

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.staticfiles import finders
from django.conf import settings

import nltk
from nltk.tokenize import word_tokenize
from gtts import gTTS
from googletrans import Translator

import mediapipe as mp

# -------------------- SAFE LEMMATIZER --------------------
from n import safe_lemmatize


# ======================================================
# PATHS & MODEL LOAD
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "live_sign",
    "alphabet_model.pkl"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "live_sign",
    "alphabet_labels.pkl"
)

WORD_MODEL_PATH = os.path.join(
    BASE_DIR,
    "live_sign",
    "word_model.pkl"
)

WORD_LABEL_PATH = os.path.join(
    BASE_DIR,
    "live_sign",
    "word_labels.pkl"
)

alphabet_model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_PATH)
word_model = joblib.load(WORD_MODEL_PATH)
word_label_encoder = joblib.load(WORD_LABEL_PATH)

print("ALPHABET MODEL LOADED FROM:", MODEL_PATH)
print("WORD MODEL LOADED FROM:", WORD_MODEL_PATH)

# ======================================================
# MEDIAPIPE HANDS (STATIC MODE - 82 FEATURES)
# ======================================================

mp_hands = mp.solutions.hands
hands_static = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

hands_word = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

FRAMES_PER_WORD = 12
VALUES_PER_FRAME = 126

word_frame_buffer = []
word_state = "IDLE"
word_last_prediction = "-"
word_last_confidence = 0.0


def extract_hand_features(landmarks):
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
    
    return np.array(features).reshape(1, -1)


def extract_word_frame_features(landmarks_list):
    """Extract 126 wrist-normalized features from hand landmarks (for word detection with 2 hands)"""
    features = []
    
    if len(landmarks_list) == 0:
        return [0.0] * VALUES_PER_FRAME
    
    for hand_landmarks in landmarks_list:
        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
        
        for lm in hand_landmarks.landmark:
            features.extend([lm.x - wrist[0], lm.y - wrist[1], lm.z - wrist[2]])
    
    if len(landmarks_list) == 1:
        features.extend([0.0] * 63)
    
    return features


# ======================================================
# BASIC VIEWS
# ======================================================

def home_view(request):
    return render(request, "home.html")


# ======================================================
# AUTH VIEWS (RESTORED)
# ======================================================

def signup_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("animation")
    else:
        form = UserCreationForm()
    return render(request, "signup.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect(request.POST.get("next", "animation"))
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("home")


# ======================================================
# TEXT → SIGN ANIMATION (OLD FEATURE – PRESERVED)
# ======================================================

@login_required(login_url="login")
def animation_view(request):
    if request.method == "POST":
        text = request.POST.get("sen", "")
        language = request.POST.get("language", "en")

        translator = Translator()
        if language == "hi":
            try:
                translated_text = translator.translate(
                    text, src="hi", dest="en"
                ).text.lower()
            except Exception:
                translated_text = text.lower()
        else:
            translated_text = text.lower()

        words = word_tokenize(translated_text)
        tagged = nltk.pos_tag(words)

        stop_words = {
            "mightn't", "re", "wasn", "wouldn", "be", "has", "that",
            "does", "shouldn", "do", "you've", "off", "for",
            "didn't", "m", "ain"
        }

        PROTECTED_WORDS = {"are", "am", "is", "was", "were", "can", "will"}

        filtered_words = [
            safe_lemmatize(w, tag)
            for w, (_, tag) in zip(words, tagged)
            if w not in stop_words or w in PROTECTED_WORDS
        ]

        final_words = []
        for w in filtered_words:
            if finders.find(f"{w}.mp4"):
                final_words.append(w)
            else:
                final_words.extend(list(w))

        audio_file = None
        if language == "hi":
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            tts = gTTS(text=text, lang="hi")
            audio_path = os.path.join(settings.MEDIA_ROOT, "output.mp3")
            tts.save(audio_path)
            audio_file = settings.MEDIA_URL + "output.mp3"

        return render(request, "animation.html", {
            "words": final_words,
            "text": text,
            "translated_text": translated_text,
            "audio": audio_file,
            "language": language
        })

    return render(request, "animation.html")


# ======================================================
# LIVE CAMERA PAGE
# ======================================================

@login_required(login_url="login")
def live_detect_view(request):
    return render(request, "live_detect.html")


# ======================================================
# RECEIVE FRAME + PREDICT (SUPPORTS ALPHABET & WORD MODES)
# ======================================================
current_detection_mode = {"mode": "alphabet"}

@csrf_exempt
def receive_frame(request):
    global word_frame_buffer, word_state, word_last_prediction, word_last_confidence, current_detection_mode
    
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    try:
        data = json.loads(request.body)
        frame = data.get("frame", "")
        mode = data.get("mode", "alphabet")
        
        if mode != current_detection_mode["mode"]:
            current_detection_mode["mode"] = mode
            word_frame_buffer = []
            word_state = "IDLE"
            word_last_prediction = "-"
            word_last_confidence = 0.0

        if not frame or "," not in frame:
            return JsonResponse({
                "label": "-",
                "confidence": 0.0
            })

        header, encoded = frame.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            return JsonResponse({
                "label": "-",
                "confidence": 0.0
            })

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if mode == "word":
            results = hands_word.process(img_rgb)
            
            if not results.multi_hand_landmarks:
                if word_state == "PREDICTED":
                    word_state = "IDLE"
                if word_state == "IDLE":
                    return JsonResponse({
                        "label": "-",
                        "confidence": 0.0,
                        "progress": 0
                    })
                word_frame_buffer.clear()
                word_state = "IDLE"
                return JsonResponse({
                    "label": word_last_prediction,
                    "confidence": word_last_confidence,
                    "is_word": True,
                    "progress": 0
                })
            
            if word_state == "IDLE":
                word_state = "COLLECTING"
                word_frame_buffer.clear()
            
            if word_state == "COLLECTING":
                frame_features = extract_word_frame_features(results.multi_hand_landmarks)
                word_frame_buffer.append(frame_features)
                
                if len(word_frame_buffer) < FRAMES_PER_WORD:
                    return JsonResponse({
                        "label": "-",
                        "confidence": 0.0,
                        "progress": len(word_frame_buffer),
                        "is_word": True
                    })
                
                if len(word_frame_buffer) > FRAMES_PER_WORD:
                    word_frame_buffer.pop(0)
                
                if len(word_frame_buffer) == FRAMES_PER_WORD:
                    X = np.array(word_frame_buffer).flatten().reshape(1, -1)
                    
                    if X.shape[1] != 1512:
                        return JsonResponse({
                            "label": "-",
                            "confidence": 0.0,
                            "is_word": True
                        })
                    
                    pred = word_model.predict(X)[0]
                    pred_word = word_label_encoder.inverse_transform([int(pred)])[0]
                    
                    if hasattr(word_model, "predict_proba"):
                        conf = float(np.max(word_model.predict_proba(X)))
                    else:
                        conf = 1.0
                    
                    word_last_prediction = pred_word
                    word_last_confidence = conf
                    word_state = "PREDICTED"
                    
                    return JsonResponse({
                        "label": pred_word,
                        "confidence": conf,
                        "is_word": True,
                        "progress": 12
                    })
            
            if word_state == "PREDICTED":
                return JsonResponse({
                    "label": word_last_prediction,
                    "confidence": word_last_confidence,
                    "is_word": True,
                    "progress": 12
                })
        
        else:
            results = hands_static.process(img_rgb)

            if not results.multi_hand_landmarks:
                return JsonResponse({
                    "label": "-",
                    "confidence": 0.0
                })

            landmarks = results.multi_hand_landmarks[0].landmark
            X = extract_hand_features(landmarks)

            if X is None or X.shape[1] == 0:
                return JsonResponse({
                    "label": "-",
                    "confidence": 0.0
                })

            pred = alphabet_model.predict(X)[0]
            pred_letter = label_encoder.inverse_transform([int(pred)])[0]

            if hasattr(alphabet_model, "predict_proba"):
                conf = float(np.max(alphabet_model.predict_proba(X)))
            else:
                conf = 1.0

            return JsonResponse({
                "label": pred_letter,
                "confidence": float(conf),
                "is_word": False
            })

    except Exception as e:
        print("receive_frame ERROR:", e)
        return JsonResponse({
            "label": "-",
            "confidence": 0.0
        })


@csrf_exempt
def reset_prediction(request):
    return JsonResponse({"status": "reset"})
