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

alphabet_model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_PATH)

print("MODEL LOADED FROM:", MODEL_PATH)

# ======================================================
# MEDIAPIPE HANDS (STATIC MODE - 82 FEATURES)
# ======================================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


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
# RECEIVE FRAME + PREDICT (STATIC MODE)
# ======================================================
@csrf_exempt
def receive_frame(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    try:
        data = json.loads(request.body)
        frame = data.get("frame", "")

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

        results = hands.process(img_rgb)

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
            "confidence": float(conf)
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
