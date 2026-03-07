# Text to Sign Language Animation - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [File Locations](#file-locations)
4. [Complete Workflow](#complete-workflow)
5. [Data Flow Diagram](#data-flow-diagram)
6. [Step-by-Step Processing](#step-by-step-processing)
7. [Example Walkthrough](#example-walkthrough)
8. [NLP Processing Details](#nlp-processing-details)
9. [Video Matching Logic](#video-matching-logic)
10. [Frontend Processing](#frontend-processing)
11. [API & Dependencies](#api--dependencies)
12. [Error Handling](#error-handling)

---

## Overview

The **Text to Sign Language Animation** module converts user-provided text (in English or Hindi) into a sequential playback of sign language video animations. This bridges communication between hearing-impaired individuals and non-sign users by visual representation.

### Key Features

- **Multi-language Support**: Accepts English and Hindi text input
- **Automatic Translation**: Hindi text is translated to English using Google Translate API
- **NLP Processing**: Tokenization, POS tagging, lemmatization, and stopword filtering
- **Video Matching**: Matches processed words to corresponding `.mp4` files in `assets/`
- **Fallback Mechanism**: Unknown words are split into individual letters
- **Speech-to-Text**: Microphone input using Web Speech API
- **Text-to-Speech**: Hindi audio output using gTTS

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TEXT → SIGN ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────┐    ┌───────────┐    ┌──────────────┐    ┌────────────────┐
    │  USER   │───▶│ DJANGO    │───▶│   NLP        │───▶│   VIDEO        │
    │  INPUT  │    │  BACKEND  │    │  PROCESSING  │    │   MATCHING     │
    └──────────┘    └───────────┘    └──────────────┘    └────────────────┘
                          │                                    │
                          ▼                                    ▼
                   ┌─────────────┐                      ┌───────────┐
                   │  TRANSLATOR │                      │  assets/  │
                   │     API     │                      │  *.mp4    │
                   └─────────────┘                      └───────────┘
                                                                    │
                                                                    ▼
    ┌──────────┐    ┌───────────┐                         ┌────────────────┐
    │ BROWSER  │◀───│ JavaScript │◀────────────────────────│  TEMPLATE      │
    │  DISPLAY │    │  PLAYER    │                         │  animation.html│
    └──────────┘    └───────────┘                         └────────────────┘
```

---

## File Locations

| File Path | Purpose |
|-----------|---------|
| `A2SL/views.py:206-260` | **Main backend logic** - handles text processing, translation, NLP, and video matching |
| `templates/animation.html` | **Frontend UI** - input form, video player, JavaScript playback logic |
| `n.py` | **NLP helper** - `safe_lemmatize()` function for word lemmatization |
| `assets/` (153 .mp4 files) | **Sign language video database** - one video per word/letter |
| `A2SL/urls.py` | URL routing configuration |
| `A2SL/settings.py` | Django settings including MEDIA_ROOT |

### Supporting Files

| File | Description |
|------|-------------|
| `templates/base.html` | Base template with common layout |
| `templates/home.html` | Home page with navigation |
| `templates/login.html` | Login form |
| `templates/signup.html` | Registration form |
| `static/mic3.png` | Microphone icon for speech input |

---

## Complete Workflow

### High-Level Process

```
USER INPUT                    BACKEND                      FRONTEND
━━━━━━━━━━━━━                ━━━━━━━━━━                    ━━━━━━━━━

1. User enters text
   (English OR Hindi)
         │
         ▼
2. Select language
         │
         ▼
3. Submit form ──────────▶ POST /animation/
         │                        │
         │                        ▼
         │                4. Language detection
         │                        │
         │                        ▼
         │                5. Translation (if Hindi)
         │                        │
         │                        ▼
         │                6. NLP Processing
         │                   - Tokenization
         │                   - POS Tagging
         │                   - Lemmatization
         │                   - Stopword filtering
         │                        │
         │                        ▼
         │                7. Video Matching
         │                   - Check word.mp4 exists
         │                   - Split unknown words
         │                        │
         │                        ▼
         │                8. Render animation.html
         │                   with word list
         │                        │
         │                        ▼
9. Display word list ◀─── Response
   and video player
         │
         ▼
10. User clicks Play
         │
         ▼
11. JavaScript plays
    videos sequentially
```

---

## Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                            USER INPUT                                          │
│                 "नमस्ते कैसे हो?" (Hindi)                                     │
└────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                         BACKEND PROCESSING                                      │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 1: LANGUAGE DETECTION                                               │  │
│  │ language = "hi" (from POST data)                                         │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                          │
│                                     ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 2: TRANSLATION (Google Translate API)                              │  │
│  │ "नमस्ते कैसे हो?" ──▶ "hello how are you"                                │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                          │
│                                     ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 3: TOKENIZATION (NLTK)                                              │  │
│  │ ["hello", "how", "are", "you"]                                          │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                          │
│                                     ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 4: POS TAGGING (NLTK)                                              │  │
│  │ [("hello","UH"), ("how","WRB"), ("are","VBP"), ("you","PRP")]          │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                          │
│                                     ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 5: LEMMATIZATION (using n.py helper)                               │  │
│  │ "are" → "are" (protected), "hello" → "hello", "you" → "you"            │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                          │
│                                     ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 6: STOPWORD FILTERING                                               │  │
│  │ Remove: "mightn't", "wasn", "be", "has", etc.                           │  │
│  │ PROTECTED: {"are", "am", "is", "was", "were", "can", "will"}           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                          │
│                                     ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 7: VIDEO MATCHING                                                   │  │
│  │ For each word:                                                           │  │
│  │   - Check if word.mp4 exists in assets/                                 │  │
│  │   - If exists: add to final_words                                       │  │
│  │   - If not: split into individual letters                                │  │
│  │                                                                         │  │
│  │ Result: ["hello.mp4", "how.mp4", "are.mp4", "you.mp4"]                  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                          │
│                                     ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ STEP 8: AUDIO GENERATION (for Hindi only)                               │  │
│  │ gTTS generates "output.mp3" for Hindi text                             │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                      RENDER animation.html                                      │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ Context variables:                                                       │  │
│  │   - words: ["hello", "how", "are", "you"]                               │  │
│  │   - text: "नमस्ते कैसे हो?"                                             │  │
│  │   - translated_text: "hello how are you"                              │  │
│  │   - audio: "/media/output.mp3" (if Hindi)                             │  │
│  │   - language: "hi"                                                     │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND DISPLAY                                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ Left Panel                                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │ Input: "नमस्ते कैसे हो?"                                         │   │  │
│  │  │ Language: Hindi                                                  │   │  │
│  │  │                                                                 │   │  │
│  │  │ Original Text: "नमस्ते कैसे हो?"                                 │   │  │
│  │  │ Translated: "hello how are you"                                 │   │  │
│  │  │                                                                 │   │  │
│  │  │ Animation Words: [hello] [how] [are] [you]                    │   │  │
│  │  │                                                                 │   │  │
│  │  │ [🔊 Audio Player]                                               │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                          │
│                                     ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ Right Panel - Video Player                                              │  │
│  │                                                                         │  │
│  │                    ┌──────────────────────┐                            │  │
│  │                    │                      │                            │  │
│  │                    │   VIDEO PLAYING...   │                            │  │
│  │                    │                      │                            │  │
│  │                    └──────────────────────┘                            │  │
│  │                                                                         │  │
│  │                    [Play/Pause]                                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │ JavaScript Sequential Playback                                          │  │
│  │   1. Get word list from <li> elements                                   │  │
│  │   2. Build video paths: "/static/hello.mp4", "/static/how.mp4"...      │  │
│  │   3. Play first video                                                    │  │
│  │   4. On 'ended' event → play next video                                  │  │
│  │   5. Highlight current word in list                                     │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Processing

### Step 1: User Input Reception

**Location:** `views.py:207-209`

```python
text = request.POST.get("sen", "")
language = request.POST.get("language", "en")
```

- Receives user text from form input field `sen`
- Receives selected language from dropdown (`en` or `hi`)
- Default language is English

### Step 2: Language Detection & Translation

**Location:** `views.py:211-220`

```python
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
```

**Why translate to English?**
- Video files in `assets/` are named in English (e.g., `Hello.mp4`, `Thank.mp4`)
- Translation normalizes input to match available video assets

### Step 3: Tokenization

**Location:** `views.py:225`

```python
words = word_tokenize(translated_text)
```

- Breaks text into individual words
- Example: `"Hello, how are you?"` → `["Hello", ",", "how", "are", "you", "?"]`
- Uses NLTK's `word_tokenize()`

### Step 4: POS Tagging

**Location:** `views.py:227`

```python
tagged = nltk.pos_tag(words)
```

- Assigns part-of-speech tags to each word
- Example: `[("Hello", "UH"), ("how", "WRB"), ("are", "VBP"), ("you", "PRP")]`
- Tags: UH (interjection), WRB (adverb), VBP (verb), PRP (pronoun)
- Required for lemmatization to determine correct word form

### Step 5: Stopword Filtering

**Location:** `views.py:229-233`

```python
stop_words = {
    "mightn't", "re", "wasn", "wouldn", "be", "has", "that",
    "does", "shouldn", "do", "you've", "off", "for",
    "didn't", "m", "ain"
}

PROTECTED_WORDS = {"are", "am", "is", "was", "were", "can", "will"}
```

- Removes common words that don't add meaning
- **Protected words** are preserved (important for sign language)

### Step 6: Lemmatization

**Location:** `views.py:238-242`

```python
filtered_words = [
    safe_lemmatize(w, tag)
    for w, (_, tag) in zip(words, tagged)
    if w not in stop_words or w in PROTECTED_WORDS
]
```

**Lemmatization Function (`n.py`):**

```python
def safe_lemmatize(word, pos_tag):
    # Protected verbs remain unchanged
    if word.lower() in AUXILIARY_VERBS:
        return word.lower()
    
    # Determine word type from POS tag
    wn_pos = get_wordnet_pos(pos_tag)
    if wn_pos:
        return lemmatizer.lemmatize(word.lower(), wn_pos)
    
    return word.lower()
```

**Examples:**
| Original | POS Tag | Lemmatized |
|----------|---------|------------|
| running | VBG | run |
| better | JJR | good |
| children | NNS | child |
| are | VBP | are (protected) |

### Step 7: Video Matching

**Location:** `views.py:244-249`

```python
final_words = []
for w in filtered_words:
    if finders.find(f"{w}.mp4"):
        final_words.append(w)
    else:
        final_words.extend(list(w))  # Split into letters
```

**Logic:**
1. For each processed word, check if `{word}.mp4` exists in `assets/`
2. If exists → add word to final list
3. If not found → split word into individual letters

**Example:**
- Input: `"Thanks"` → `Thanks.mp4` exists → `["Thanks"]`
- Input: `"NLP"` → No `NLP.mp4` → `["N", "L", "P"]`

### Step 8: Audio Generation (Hindi)

**Location:** `views.py:251-256`

```python
if language == "hi":
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    tts = gTTS(text=text, lang="hi")
    audio_path = os.path.join(settings.MEDIA_ROOT, "output.mp3")
    tts.save(audio_path)
    audio_file = settings.MEDIA_URL + "output.mp3"
```

- Uses Google Text-to-Speech (gTTS) for Hindi
- Saves audio to `media/output.mp3`
- Displayed in HTML audio player

### Step 9: Render Response

**Location:** `views.py:260-267`

```python
return render(request, "animation.html", {
    "words": final_words,
    "text": text,
    "translated_text": translated_text,
    "audio": audio_file,
    "language": language
})
```

---

## Example Walkthrough

### Example 1: Hindi Input

**Input:**
```
Text: "नमस्ते कैसे हो"
Language: Hindi
```

**Processing Steps:**

| Step | Input | Output |
|------|-------|--------|
| 1. Translation | "नमस्ते कैसे हो" | "hello how are you" |
| 2. Tokenization | "hello how are you" | ["hello", "how", "are", "you"] |
| 3. POS Tagging | ["hello", "how", "are", "you"] | [("hello","UH"), ("how","WRB"), ("are","VBP"), ("you","PRP")] |
| 4. Lemmatization | All words | ["hello", "how", "are", "you"] |
| 5. Stopword Filter | No stopwords | ["hello", "how", "are", "you"] |
| 6. Video Match | Check each word | [Hello.mp4, how.mp4, are.mp4, you.mp4] |

**Final Output:**
- **words:** `["hello", "how", "are", "you"]`
- **translated_text:** `"hello how are you"`
- **audio:** `/media/output.mp3`

---

### Example 2: English Input with Unknown Words

**Input:**
```
Text: "I love AI"
Language: English
```

**Processing Steps:**

| Step | Input | Output |
|------|-------|--------|
| 1. Translation | Skip (already English) | "i love ai" |
| 2. Tokenization | "i love ai" | ["i", "love", "ai"] |
| 3. POS Tagging | ["i", "love", "ai"] | [("i","PRP"), ("love","VBP"), ("ai","NN")] |
| 4. Lemmatization | ["i", "love", "ai"] | ["i", "love", "ai"] |
| 5. Stopword Filter | "i" is not in stop_words | ["i", "love", "ai"] |
| 6. Video Match | "i" → i.mp4 ✓<br>"love" → love.mp4 ✗<br>"ai" → ai.mp4 ✗ | ["i", "l", "o", "v", "e", "a", "i"] |

**Note:** 
- "love" is split into letters because `love.mp4` doesn't exist
- "ai" is split into "a" and "i"

---

### Example 3: Sentence with Stopwords

**Input:**
```
Text: "The weather is good today"
Language: English
```

**Processing:**

| Word | Stopword? | Video Exists? | Output |
|------|-----------|---------------|--------|
| The | ✓ (removed) | - | - |
| weather | ✗ | ✗ | "w", "e", "a", "t", "h", "e", "r" |
| is | ✓ but PROTECTED | ✗ | "i", "s" |
| good | ✗ | ✓ | "good" |
| today | ✗ | ✗ | "t", "o", "d", "a", "y" |

**Final words:** `["w", "e", "a", "t", "h", "e", "r", "i", "s", "good", "t", "o", "d", "a", "y"]`

---

## NLP Processing Details

### Tokenization

Uses NLTK's `word_tokenize()` which handles:
- Contractions (won't, can't)
- Punctuation (., ,, ?)
- Special characters

```python
from nltk.tokenize import word_tokenize

text = "Hello, how's it going?"
tokens = word_tokenize(text)
# Output: ['Hello', ',', 'how', "'s", 'it', 'going', '?']
```

### POS Tagging

Part-of-Speech tags used for lemmatization:

| Tag | Meaning | Example |
|-----|---------|---------|
| VB | Verb (base) | "run" |
| VBP | Verb (non-3rd person) | "run" |
| VBZ | Verb (3rd person) | "runs" |
| VBG | Verb (gerund) | "running" |
| VBN | Verb (past participle) | "run" |
| NN | Noun (singular) | "cat" |
| NNS | Noun (plural) | "cats" |
| JJ | Adjective | "big" |
| JJR | Adjective (comparative) | "bigger" |
| JJS | Adjective (superlative) | "biggest" |
| PRP | Personal pronoun | "I", "you" |
| UH | Interjection | "hello" |

### Lemmatization

Uses WordNet lemmatizer with POS-aware processing:

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

# With POS tag
lemmatizer.lemmatize("running", wordnet.VERB)  # "run"
lemmatizer.lemmatize("better", wordnet.ADJ)    # "good"
lemmatizer.lemmatize("children", wordnet.NOUN) # "child"
```

### Protected Words

Certain words must not be lemmatized because they are essential in sign language:

```python
PROTECTED_WORDS = {"are", "am", "is", "was", "were", "can", "will"}
```

Without protection: "are" → "be" (loses meaning)
With protection: "are" → "are" (correct)

---

## Video Matching Logic

### File Lookup Process

```python
def match_word_to_video(word, finders):
    # Convert to lowercase for matching
    word_lower = word.lower()
    
    # Check if video file exists
    if finders.find(f"{word_lower}.mp4"):
        return [word_lower]
    else:
        # Split into individual letters
        return list(word_lower)
```

### Available Video Files

The `assets/` folder contains 153 `.mp4` files:

**Alphabets (26):**
```
A.mp4, B.mp4, C.mp4, ... Z.mp4
```

**Common Words (127):**
```
Hello.mp4, Thank.mp4, Good.mp4, Bad.mp4,
How.mp4, What.mp4, Where.mp4, When.mp4,
I.mp4, You.mp4, We.mp4, They.mp4,
Me.mp4, My.mp4, Your.mp4, His.mp4, Her.mp4,
Come.mp4, Go.mp4, Eat.mp4, Drink.mp4,
Yes.mp4, No.mp4, Please.mp4, Sorry.mp4,
...
```

### Matching Algorithm

```
For each word in filtered_words:
    │
    ├──▶ Does word.mp4 exist?
    │       │
    │       ├── YES ──▶ Add word to final_words
    │       │
    │       └── NO ──▶ Split into letters
    │                       │
    │                       └──▶ Add each letter to final_words
    │
    └──▶ Continue to next word
```

---

## Frontend Processing

### Animation Template

**Location:** `templates/animation.html`

The template receives the processed word list and renders:

1. **Input Form** - Text input + language selector + microphone button
2. **Results Table** - Original text, translated text, animation words
3. **Video Player** - HTML5 video element for playback
4. **Audio Player** - For Hindi TTS output
5. **Spectrogram** - Real-time audio visualization

### JavaScript Playback Logic

**Location:** `templates/animation.html:350-404`

```javascript
function play() {
    // 1. Collect all words from <li> elements
    var videos = document.getElementById("list").getElementsByTagName("li");
    
    // 2. Build video source paths
    var videoSource = [];
    for (var j = 0; j < videos.length; j++) {
        videoSource[j] = "/static/" + videos[j].innerHTML + ".mp4";
    }
    
    // 3. Play first video
    videoPlay(0);
    
    // 4. Sequential playback handler
    function myHandler() {
        i++;
        if (i == videoCount) {
            document.getElementById("videoPlayer").pause();
        } else {
            videoPlay(i);  // Play next video
        }
    }
}

function videoPlay(videoNum) {
    // Highlight current word
    document.getElementById("list").getElementsByTagName("li")[videoNum]
        .style.color = "#09edc7";
    
    // Set video source and play
    document.getElementById("videoPlayer").setAttribute("src", videoSource[videoNum]);
    document.getElementById("videoPlayer").load();
    document.getElementById("videoPlayer").play();
}
```

### Speech-to-Text

**Location:** `templates/animation.html:324-348`

Uses Web Speech API for microphone input:

```javascript
function record() {
    var recognition = new webkitSpeechRecognition();
    var lang = document.getElementById('language').value;
    
    recognition.lang = lang === 'hi' ? 'hi-IN' : 'en-IN';
    
    recognition.onresult = function(event) {
        document.getElementById('speechToText').value = 
            event.results[0][0].transcript;
    };
    
    recognition.start();
}
```

### Spectrogram Visualization

**Location:** `templates/animation.html:426-476`

Real-time audio visualization during speech input using Web Audio API:

```javascript
function startSpectrogram() {
    audioContext = new AudioContext();
    analyser = audioContext.createAnalyser();
    
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            source = audioContext.createMediaStreamSource(stream);
            source.connect(analyser);
            
            function draw() {
                analyser.getByteFrequencyData(dataArray);
                // Draw frequency bars on canvas
            }
        });
}
```

---

## API & Dependencies

### Python Libraries

| Library | Purpose |
|---------|---------|
| `django` | Web framework |
| `nltk` | NLP processing (tokenization, POS, lemmatization) |
| `googletrans` | Google Translate API wrapper |
| `gTTS` | Google Text-to-Speech |
| `Pillow` | Image processing (if needed) |

### NLTK Data Downloads

```python
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
```

### JavaScript APIs

| API | Purpose |
|-----|---------|
| `webkitSpeechRecognition` | Speech-to-text |
| `Web Audio API` | Audio visualization |
| `HTML5 Video` | Video playback |

### External Services

| Service | Purpose | Rate Limit |
|---------|---------|-------------|
| Google Translate API | Hindi → English | Varies by plan |
| Google Text-to-Speech | Hindi audio generation | Varies by plan |

---

## Error Handling

### Translation Errors

```python
try:
    translated_text = translator.translate(
        text, src="hi", dest="en"
    ).text.lower()
except Exception:
    translated_text = text.lower()  # Fallback to original
```

If translation fails, uses original text (English).

### Missing Video Files

```python
if finders.find(f"{w}.mp4"):
    final_words.append(w)
else:
    final_words.extend(list(w))  # Split into letters
```

Unknown words are split into individual letters as fallback.

### Empty Input

The form requires at least some text. Empty submissions result in empty word list.

### Microphone Errors

```javascript
recognition.onerror = function(event) {
    console.error('Speech recognition error:', event.error);
    stopSpectrogram();
};
```

---

## URL Configuration

**Location:** `A2SL/urls.py`

```python
path('animation/', views.animation_view, name='animation'),
```

**Full URL:** `http://127.0.0.1:8000/animation/`

---

## Session Flow

```
1. User visits / → Home page
   
2. User clicks "Express" or "Login" first
   - If not logged in → Redirect to /login/
   
3. After login → Redirect to /animation/
   
4. User enters text → POST to /animation/
   
5. Backend processes → Returns animation.html
   
6. User clicks "Play" → Videos play sequentially
   
7. User can:
   - Enter new text
   - Use microphone
   - Change language
   - Logout
```

---

## Summary

The Text to Sign Language Animation module is a complete pipeline that:

1. **Receives** user text input (English or Hindi)
2. **Translates** Hindi to English if needed
3. **Processes** text using NLP (tokenize, POS tag, lemmatize, filter)
4. **Matches** each word to available video files
5. **Falls back** to letter-by-letter for unknown words
6. **Renders** HTML with video list
7. **Plays** videos sequentially in browser

This creates an accessible communication bridge for hearing-impaired individuals.
