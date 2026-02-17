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

# ======================================================
# MEDIAPIPE HANDS (63 FEATURES)
# ======================================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)


def extract_hand_features(img):
    """
    Returns (1, 63) numpy array or None
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    landmarks = result.multi_hand_landmarks[0].landmark
    features = []

    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])

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
# RECEIVE FRAME + PREDICT (FINAL & FIXED)
# ======================================================
@csrf_exempt
def receive_frame(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    try:
        data = json.loads(request.body)
        frame = data.get("frame", "")

        # ---------- 1. Validate frame ----------
        if not frame or "," not in frame:
            return JsonResponse({
                "label": "-",
                "confidence": 0.0
            })

        # ---------- 2. Decode base64 ----------
        header, encoded = frame.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)

        # ---------- 3. Decode image ----------
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            return JsonResponse({
                "label": "-",
                "confidence": 0.0
            })

        # ---------- 4. Convert to RGB ----------
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---------- 5. MediaPipe Hands ----------
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return JsonResponse({
                "label": "-",
                "confidence": 0.0
            })

        # ---------- 6. Extract 63 features ----------
        hand_landmarks = results.multi_hand_landmarks[0]
        features = []

        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])

        if len(features) != 63:
            return JsonResponse({
                "label": "-",
                "confidence": 0.0
            })

        X = np.array(features).reshape(1, -1)

        # ---------- 7. Predict ----------
        pred = alphabet_model.predict(X)[0]
        pred_letter = label_encoder.inverse_transform([int(pred)])[0]#convert number into alphabet

        if hasattr(alphabet_model, "predict_proba"):
            conf = float(np.max(alphabet_model.predict_proba(X)))
        else:
            conf = 1.0

        # ---------- 8. Return SAFE JSON ----------
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
