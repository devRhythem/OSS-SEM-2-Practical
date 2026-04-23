from sklearn.datasets import load_breast_cancer  # Dataset
from sklearn.model_selection import train_test_split  # Split train/test
from sklearn.linear_model import LogisticRegression  # Linear classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt

# 1. Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data       # Features (measurements of cell nuclei)
y = data.target     # Target (0 = malignant, 1 = benign)

# 2. Split dataset into train and test
# test_size=0.3 → 30% test data, random_state=42 → reproducible results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize Logistic Regression model
# solver='liblinear' works well for small datasets and binary classification
model = LogisticRegression(solver='liblinear')

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions on test set
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification report (summary of precision, recall, F1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. ROC Curve and AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability scores
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)  # Compute ROC curve
auc = roc_auc_score(y_test, y_pred_proba)  # Area Under Curve

print("\nAUC Score:", auc)

# Plot ROC curve
plt.plot(fpr, tpr, label="Logistic Regression (AUC = %.2f)" % auc)
plt.plot([0,1], [0,1], linestyle="--", color="gray")  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
import seaborn as sns
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Benign (0)', 'Malignant (1)'],
            yticklabels=['Benign (0)', 'Malignant (1)'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()
