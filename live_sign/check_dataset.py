import numpy as np
import os
from collections import Counter

print("\n========== DATASET CHECK ==========\n")

ALPHA_DIR = "live_sign/data_alphabets_static"

if os.path.exists(ALPHA_DIR):
    try:
        X_alpha = np.load(os.path.join(ALPHA_DIR, "X.npy"))
        y_alpha = np.load(os.path.join(ALPHA_DIR, "y.npy"))

        print("ALPHABET DATASET")
        print("----------------")
        print("Total samples:", len(y_alpha))
        print("Feature shape:", X_alpha.shape)

        counts = Counter(y_alpha)
        for k in sorted(counts):
            print(f"{k}: {counts[k]}")
        print()
    except:
        print("❌ Alphabet dataset files missing\n")
else:
    print("❌ Alphabet dataset folder not found\n")

WORD_DIR = "live_sign/data_words"

if os.path.exists(WORD_DIR):
    try:
        X_word = np.load(os.path.join(WORD_DIR, "X.npy"))
        y_word = np.load(os.path.join(WORD_DIR, "y.npy"))

        print("WORD DATASET")
        print("------------")
        print("Total samples:", len(y_word))
        print("Feature shape:", X_word.shape)

        counts = Counter(y_word)
        for k in sorted(counts):
            print(f"{k}: {counts[k]}")
        print()
    except:
        print("❌ Word dataset files missing\n")
else:
    print("❌ Word dataset folder not found\n")

print("========== CHECK COMPLETE ==========\n")
