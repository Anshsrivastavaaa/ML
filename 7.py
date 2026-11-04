# ============================================================
# 7.py — Support Vector Machine (SVM) Classification
#         Comparing Linear vs RBF kernels on the Iris dataset
# ============================================================

# --- Import libraries ---
from sklearn.datasets import load_iris                     # Classic Iris flower dataset (150 samples, 4 features, 3 classes)
from sklearn.model_selection import train_test_split       # For splitting data into train/test sets
from sklearn.svm import SVC                                # Support Vector Classifier
from sklearn.preprocessing import StandardScaler           # For z-score feature normalization
from sklearn.pipeline import make_pipeline                 # For chaining preprocessing + model into one pipeline
from sklearn.metrics import accuracy_score, classification_report  # For evaluating model performance

# --- Load the Iris dataset ---
# 150 samples of iris flowers with 4 features each:
# sepal length, sepal width, petal length, petal width
# 3 target classes: Setosa (0), Versicolor (1), Virginica (2)
data = load_iris()
X, y = data.data, data.target

# --- Split into training and testing sets ---
# 80% train, 20% test; stratify=y ensures each class is proportionally represented
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Linear SVM ---
# Create a pipeline that first scales features, then applies Linear SVM
# Linear kernel: finds a straight hyperplane to separate classes
# Pipeline ensures scaling is applied consistently to train and test data
linear_model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
linear_model.fit(X_train, y_train)          # Train the pipeline
linear_pred = linear_model.predict(X_test)  # Predict on test data

# --- Nonlinear SVM (RBF kernel) ---
# RBF (Radial Basis Function) kernel: maps data to higher dimensions
# Can find curved/nonlinear decision boundaries for complex class separations
rbf_model = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
rbf_model.fit(X_train, y_train)             # Train the pipeline
rbf_pred = rbf_model.predict(X_test)        # Predict on test data

# --- Compare performance ---
# Print overall accuracy for both models
print("Linear SVM Accuracy:",
      accuracy_score(y_test, linear_pred))

print("RBF SVM Accuracy:",
      accuracy_score(y_test, rbf_pred))

# Print detailed per-class metrics: precision, recall, f1-score, and support
print("\nLinear SVM Classification Report:")
print(classification_report(y_test, linear_pred))

print("RBF SVM Classification Report:")
print(classification_report(y_test, rbf_pred))
