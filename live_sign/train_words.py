"""
train_words.py

Purpose:
    Trains an SVM (Support Vector Machine) classifier on dynamic word sign data.
    
    This script is similar to train_alphabets_static.py but works with
    multi-frame gesture data (12 frames × 126 features = 1512 features per sample).
    
    Pipeline:
    1. Load recorded word data (features + labels)
    2. Encode labels using LabelEncoder
    3. Split data into train/test sets (80/20 split)
    4. Train SVM 
    5. Evaluate model and generate confusion matrix + classification report
    6. Save trained model and label encoder

Input:
    - live_sign/data_words_v2/X.npy: Feature array (N x 1512)
    - live_sign/data_words_v2/y.npy: Label array (N,)

Output:
    - live_sign/word_model.pkl: Trained SVM model
    - live_sign/word_labels.pkl: LabelEncoder for inverse transform
    - live_sign/word_confusion_matrix.png: Visualization
    - live_sign/word_classification_report.txt: Detailed metrics
"""

import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# LOAD DATA
# ============================================================================

DATA_DIR = "live_sign/data_words_v2"

# Load features (X) and labels (y) from numpy files
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

print("Word data loaded:")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================================
# LABEL ENCODING
# ============================================================================

# Convert string labels (HOW, YOU, THANK, ...) to integer labels (0, 1, 2, ...)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

# Split data: 80% training, 20% testing
# stratify ensures equal distribution of each class in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# ============================================================================
# MODEL TRAINING
# ============================================================================

# Create SVM classifier with RBF (Radial Basis Function) kernel
# probability=True: Enable probability estimates for confidence scores
# Note: Unlike train_alphabets_static.py, no StandardScaler is used here
#       (the data may already be normalized from the recording process)
model = SVC(kernel="rbf", probability=True)

# Train the model
model.fit(X_train, y_train)


# ============================================================================
# EVALUATION
# ============================================================================

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Word Model Accuracy: {accuracy * 100:.2f}%")


# ============================================================================
# CONFUSION MATRIX & CLASSIFICATION REPORT
# ============================================================================

print("\n" + "="*60)
print("GENERATING CLASSIFICATION REPORT & CONFUSION MATRIX")
print("="*60)

# Get class names from encoder (e.g., ['HOW', 'YOU', 'THANK', ...])
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

print("\n=== WORST PERFORMING WORDS (Lowest F1 Score) ===")
worst_5 = sorted_classes[:5]
for word, f1 in worst_5:
    precision = report[word]['precision']
    recall = report[word]['recall']
    support = report[word]['support']
    print(f"  {word}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, Support={support}")

print("\n=== BEST PERFORMING WORDS (Highest F1 Score) ===")
best_5 = sorted_classes[-5:][::-1]
for word, f1 in best_5:
    print(f"  {word}: F1={f1:.3f}")


# ============================================================================
# PLOT CONFUSION MATRIX
# ============================================================================

# Determine figure size based on number of classes
# Each class needs ~0.8 inches, minimum 10 inches
num_classes = len(class_names)
fig_size = max(10, num_classes * 0.8)

# Create heatmap visualization of confusion matrix
plt.figure(figsize=(fig_size, fig_size * 0.9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 8})
plt.title('Word Detection Confusion Matrix\n(Dynamic Mode)', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('live_sign/word_confusion_matrix.png', dpi=150)
plt.close()
print("\nConfusion matrix saved to: live_sign/word_confusion_matrix.png")


# ============================================================================
# SAVE CLASSIFICATION REPORT
# ============================================================================

report_path = "live_sign/word_classification_report.txt"
with open(report_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("WORD DETECTION - CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write(f"Total Test Samples: {len(y_test)}\n")
    f.write(f"Number of Classes: {len(class_names)}\n\n")
    f.write("-"*60 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("-"*60 + "\n\n")
    f.write(report_str)
    f.write("\n\n")
    f.write("-"*60 + "\n")
    f.write("WORST PERFORMING WORDS (Bottom 5)\n")
    f.write("-"*60 + "\n")
    for word, f1 in worst_5:
        precision = report[word]['precision']
        recall = report[word]['recall']
        support = report[word]['support']
        f.write(f"  {word}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, Support={support}\n")
    f.write("\n")
    f.write("-"*60 + "\n")
    f.write("BEST PERFORMING WORDS (Top 5)\n")
    f.write("-"*60 + "\n")
    for word, f1 in best_5:
        f.write(f"  {word}: F1={f1:.3f}\n")

print(f"Classification report saved to: {report_path}")


# ============================================================================
# SAVE MODEL
# ============================================================================

# Save trained SVM model
joblib.dump(model, "live_sign/word_model.pkl")

# Save label encoder for inverse transform (predictions -> words)
joblib.dump(encoder, "live_sign/word_labels.pkl")

print("\n✅ Word model saved successfully")
