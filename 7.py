from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Linear SVM
linear_model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# Nonlinear SVM (RBF kernel)
rbf_model = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
rbf_model.fit(X_train, y_train)
rbf_pred = rbf_model.predict(X_test)

# Compare performance
print("Linear SVM Accuracy:",
      accuracy_score(y_test, linear_pred))

print("RBF SVM Accuracy:",
      accuracy_score(y_test, rbf_pred))

print("\nLinear SVM Classification Report:")
print(classification_report(y_test, linear_pred))

print("RBF SVM Classification Report:")
print(classification_report(y_test, rbf_pred))


 
