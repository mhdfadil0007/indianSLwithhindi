import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATA_DIR = "live_sign/data_alphabets_static"

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

print("Static alphabet data loaded:", X.shape, y.shape)

def augment_flip(X, y):
    n_landmarks = 63
    n_distances = 5 + 10 + 4
    
    X_flipped = X.copy()
    for i in range(len(X_flipped)):
        for j in range(21):
            idx = j * 3
            X_flipped[i, idx] = 1.0 - X_flipped[i, idx]
            X_flipped[i, idx + 2] = -X_flipped[i, idx + 2]
        
        dist_start = n_landmarks
        for k in range(n_distances):
            X_flipped[i, dist_start + k] = X_flipped[i, dist_start + k]
    
    return np.vstack([X, X_flipped]), np.concatenate([y, y])

print("Augmenting with horizontal flip...")
X_aug, y_aug = augment_flip(X, y)
print(f"Augmented data: {X_aug.shape}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_aug)

X_train, X_test, y_train, y_test = train_test_split(
    X_aug,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Static Alphabet Accuracy: {acc * 100:.2f}%")

joblib.dump(model, "live_sign/alphabet_model.pkl")
joblib.dump(encoder, "live_sign/alphabet_labels.pkl")

print("✅ STATIC alphabet model saved")
print("Unique labels:", np.unique(y))

