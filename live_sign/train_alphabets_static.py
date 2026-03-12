"""
train_alphabets_static.py

Purpose:
    Trains an SVM (Support Vector Machine) classifier on static alphabet hand sign data.
    
    Pipeline:
    1. Load recorded alphabet data (features + labels)
    2. Apply data augmentation (horizontal flip) to increase training samples
    3. Encode labels using LabelEncoder
    4. Split data into train/test sets (80/20 split)
    5. Train SVM 
    6. Evaluate model and generate confusion matrix + classification report
    7. Save trained model and label encoder

Input:
    - live_sign/data_alphabets_static/X.npy: Feature array
    - live_sign/data_alphabets_static/y.npy: Label array

Output:
    - live_sign/alphabet_model.pkl: Trained SVM model
    - live_sign/alphabet_labels.pkl: LabelEncoder for inverse transform
    - live_sign/alphabet_confusion_matrix.png: Visualization
    - live_sign/alphabet_classification_report.txt: Detailed metrics
"""

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


# ============================================================================
# LOAD DATA
# ============================================================================

DATA_DIR = "live_sign/data_alphabets_static"

# Load features (X) and labels (y) from numpy files
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

print("Static alphabet data loaded:", X.shape, y.shape)


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_flip(X, y):
    """
    Performs data augmentation using horizontal flip.
    
    For hand signs, a horizontal flip represents the mirror image,
    which is a valid variation (left hand becomes right hand).
    
    
    """
    n_landmarks = 63  # 21 landmarks × 3 coordinates
    n_distances = 5 + 10 + 4  # fingertip distances + inter-fingertip + angles
    
    X_flipped = X.copy()
    for i in range(len(X_flipped)):
        # Flip x-coordinates (21 landmarks × 3 = 63 values starting at index 0)
        for j in range(21):
            idx = j * 3
            X_flipped[i, idx] = 1.0 - X_flipped[i, idx]
            # Flip z-coordinate (depth)
            X_flipped[i, idx + 2] = -X_flipped[i, idx + 2]
        
        # Distances and angles remain the same after horizontal flip
        dist_start = n_landmarks
        for k in range(n_distances):
            X_flipped[i, dist_start + k] = X_flipped[i, dist_start + k]
    
    # Stack original and augmented data
    return np.vstack([X, X_flipped]), np.concatenate([y, y])

print("Augmenting with horizontal flip...")
X_aug, y_aug = augment_flip(X, y)
print(f"Augmented data: {X_aug.shape}")


# ============================================================================
# LABEL ENCODING
# ============================================================================

# Convert string labels (A, B, C, ...) to integer labels (0, 1, 2, ...)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_aug)


# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

# Split data: 80% training, 20% testing
# stratify ensures equal distribution of each class in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_aug,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# ============================================================================
# MODEL TRAINING
# ============================================================================

# Create sklearn pipeline with:
# 1. StandardScaler: Normalizes features to have mean=0, std=1
# 2. SVC with RBF kernel: Non-linear classifier
#    - C=10: Regularization parameter (higher = stricter margin)
#    - gamma="scale": Automatic gamma based on features
#    - probability=True: Enable probability estimates
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
])

# Train the model
model.fit(X_train, y_train)


# ============================================================================
# EVALUATION
# ============================================================================

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)

print(f"Static Alphabet Accuracy: {acc * 100:.2f}%")


# ============================================================================
# CONFUSION MATRIX & CLASSIFICATION REPORT
# ============================================================================

print("\n" + "="*60)
print("GENERATING CLASSIFICATION REPORT & CONFUSION MATRIX")
print("="*60)

# Get class names from encoder (e.g., ['A', 'B', 'C', ...])
class_names = encoder.classes_

# Generate confusion matrix: rows = true labels, columns = predicted
cm = confusion_matrix(y_test, y_pred)

# Generate classification report as dictionary and string
# Includes: precision, recall, f1-score, support for each class
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


# ============================================================================
# PLOT CONFUSION MATRIX
# ============================================================================

# Create heatmap visualization of confusion matrix
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


# ============================================================================
# SAVE CLASSIFICATION REPORT
# ============================================================================

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


# ============================================================================
# SAVE MODEL
# ============================================================================

# Save trained model (includes scaler + SVM)
joblib.dump(model, "live_sign/alphabet_model.pkl")

# Save label encoder for inverse transform (predictions -> letters)
joblib.dump(encoder, "live_sign/alphabet_labels.pkl")

print("\n✅ STATIC alphabet model saved")
print("Unique labels:", np.unique(y))
