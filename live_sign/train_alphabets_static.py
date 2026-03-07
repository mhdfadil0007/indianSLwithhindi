import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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

# ===================== CONFUSION MATRIX & METRICS =====================
print("\n" + "="*60)
print("GENERATING CLASSIFICATION REPORT & CONFUSION MATRIX")
print("="*60)

# Get class names
class_names = encoder.classes_

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Classification report
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
report_str = classification_report(y_test, y_pred, target_names=class_names)

print("\n=== Classification Report ===")
print(report_str)

# Identify worst performing classes (lowest F1 score)
class_f1_scores = {cls: report[cls]['f1-score'] for cls in class_names}
sorted_classes = sorted(class_f1_scores.items(), key=lambda x: x[1])

print("\n=== WORST PERFORMING LETTERS (Lowest F1 Score) ===")
worst_5 = sorted_classes[:5]
for letter, f1 in worst_5:
    precision = report[letter]['precision']
    recall = report[letter]['recall']
    support = report[letter]['support']
    print(f"  {letter}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, Support={support}")

print("\n=== BEST PERFORMING LETTERS (Highest F1 Score) ===")
best_5 = sorted_classes[-5:][::-1]
for letter, f1 in best_5:
    print(f"  {letter}: F1={f1:.3f}")

# Plot confusion matrix
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 8})
plt.title('Alphabet Detection Confusion Matrix\n(Static Mode)', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('live_sign/alphabet_confusion_matrix.png', dpi=150)
plt.close()
print("\nConfusion matrix saved to: live_sign/alphabet_confusion_matrix.png")

# Save classification report to text file
report_path = "live_sign/alphabet_classification_report.txt"
with open(report_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("ALPHABET DETECTION - CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Overall Accuracy: {acc * 100:.2f}%\n\n")
    f.write(f"Total Test Samples: {len(y_test)}\n")
    f.write(f"Number of Classes: {len(class_names)}\n\n")
    f.write("-"*60 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("-"*60 + "\n\n")
    f.write(report_str)
    f.write("\n\n")
    f.write("-"*60 + "\n")
    f.write("WORST PERFORMING LETTERS (Bottom 5)\n")
    f.write("-"*60 + "\n")
    for letter, f1 in worst_5:
        precision = report[letter]['precision']
        recall = report[letter]['recall']
        support = report[letter]['support']
        f.write(f"  {letter}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, Support={support}\n")
    f.write("\n")
    f.write("-"*60 + "\n")
    f.write("BEST PERFORMING LETTERS (Top 5)\n")
    f.write("-"*60 + "\n")
    for letter, f1 in best_5:
        f.write(f"  {letter}: F1={f1:.3f}\n")

print(f"Classification report saved to: {report_path}")

# Save model
joblib.dump(model, "live_sign/alphabet_model.pkl")
joblib.dump(encoder, "live_sign/alphabet_labels.pkl")

print("\n✅ STATIC alphabet model saved")
print("Unique labels:", np.unique(y))

