import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = "live_sign/data_words_v2"

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

print("Word data loaded:")
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

print(f"Word Model Accuracy: {accuracy * 100:.2f}%")

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

# Plot confusion matrix (smaller figure for words)
num_classes = len(class_names)
fig_size = max(10, num_classes * 0.8)

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

# Save classification report to text file
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

# Save model
joblib.dump(model, "live_sign/word_model.pkl")
joblib.dump(encoder, "live_sign/word_labels.pkl")

print("\n✅ Word model saved successfully")
