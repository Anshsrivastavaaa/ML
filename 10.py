# ============================================================
# 10.py — Ensemble Learning: Logistic Regression vs Random Forest
#          on the Breast Cancer dataset
# ============================================================

# --- Import libraries ---
from sklearn.datasets import load_breast_cancer             # Breast cancer classification dataset
from sklearn.model_selection import train_test_split        # For splitting data into train/test sets
from sklearn.preprocessing import StandardScaler            # For z-score feature normalization
from sklearn.pipeline import make_pipeline                  # For chaining scaler + model together
from sklearn.linear_model import LogisticRegression         # Logistic Regression (baseline model)
from sklearn.ensemble import RandomForestClassifier         # Random Forest ensemble classifier
from sklearn.metrics import accuracy_score, classification_report  # Evaluation metrics

# --- Load the Breast Cancer dataset ---
# 569 samples, 30 numerical features, binary target (0=malignant, 1=benign)
data = load_breast_cancer()
X, y = data.data, data.target

# --- Split into training and testing sets ---
# 80% train, 20% test; stratify=y preserves the class ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Baseline model: Logistic Regression ---
# Pipeline: first scales features (StandardScaler), then applies Logistic Regression
# max_iter=2000: allow enough iterations for the optimizer to converge
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
lr.fit(X_train, y_train)               # Train the pipeline on training data
lr_pred = lr.predict(X_test)            # Predict on test data

# --- Ensemble model: Random Forest ---
# Random Forest builds 100 decision trees and aggregates their predictions (majority vote)
# Each tree is trained on a random subset of data and features (bagging + feature randomness)
# No feature scaling needed — tree-based models are scale-invariant
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)               # Train the forest on training data
rf_pred = rf.predict(X_test)            # Predict on test data

# --- Compare both models ---
# Print overall accuracy for each model
print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred))

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))

# Print detailed per-class metrics: precision, recall, f1-score, support
print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_pred))

print("Random Forest Report:")
print(classification_report(y_test, rf_pred))
