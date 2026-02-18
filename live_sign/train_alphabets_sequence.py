import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


DATA_DIR = "live_sign/data_alphabets_sequence"

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

print("Alphabet sequence data loaded:")
print("X shape:", X.shape)
print("y shape:", y.shape)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Alphabet Sequence Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "live_sign/alphabet_model.pkl")
joblib.dump(encoder, "live_sign/alphabet_labels.pkl")

print("✅ Alphabet sequence model saved successfully")
