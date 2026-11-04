# ============================================================
# 6.py — Logistic Regression for Heart Disease Classification
#         with Confusion Matrix, Precision, Recall, F1 Score
# ============================================================

# --- Import libraries ---
import numpy as np                                        # For numerical operations
import pandas as pd                                       # For data manipulation with DataFrames
import matplotlib.pyplot as plt                           # For plotting visualizations
from sklearn.preprocessing import LabelEncoder            # For converting categorical text to numbers
from sklearn.model_selection import train_test_split      # For splitting data into train/test sets
from sklearn.linear_model import LogisticRegression       # Logistic Regression classifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score  # Evaluation metrics

# --- Load the dataset ---
# Read the heart disease dataset from CSV
df = pd.read_csv('heart_disease.csv')

# --- Exploratory Data Analysis ---
# Display the first 5 rows to understand data structure
df.head()

# Show column types, non-null counts, and memory usage
df.info()

# Display statistical summary of numerical columns
df.describe()

# --- Handle missing values ---
# Fill missing numerical values with the column median (robust to outliers)
df = df.fillna(df.median(numeric_only=True))

# Verify all missing values have been handled
print(df.isnull().sum())

# --- Encode categorical variables ---
# Convert all text/object columns to numerical values using LabelEncoder
# Each unique category gets a unique integer (e.g., 'Yes'=1, 'No'=0)
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# --- Prepare features and target ---
# Features: all columns except the target 'Heart Disease Status'
x = df.drop(['Heart Disease Status'], axis=1)

# Target: binary label indicating presence of heart disease
y = df['Heart Disease Status']

# --- Split into training and testing sets ---
# 80% train, 20% test; stratify=y ensures both sets have the same class proportions
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train the Logistic Regression model ---
# max_iter=5000: allow more iterations for convergence
# class_weight='balanced': automatically adjusts weights for imbalanced classes
model = LogisticRegression(max_iter=5000, class_weight='balanced')
model.fit(x_train, y_train)

# --- Generate predictions and confusion matrix ---
# Predict heart disease status on the test set
model_predict = model.predict(x_test)

# Compute the confusion matrix:
#   [[True Negatives,  False Positives],
#    [False Negatives, True Positives]]
cm = confusion_matrix(y_test, model_predict)

print("Confusion Matrix:")
print(cm)

# --- Visualize the confusion matrix ---
# Display the confusion matrix as a color-coded heatmap
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

# Set tick labels to meaningful class names
plt.xticks([0, 1], ["No", "Yes"])
plt.yticks([0, 1], ["No", "Yes"])

# Annotate each cell with its count value
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

# Display the plot
plt.show()

# --- Calculate and display classification metrics ---
# Precision: Of all predicted positives, how many were actually positive?
# High precision = few false positives
precision = precision_score(y_test, model_predict)
print("Precision:", precision)

# Recall (Sensitivity): Of all actual positives, how many were correctly identified?
# High recall = few false negatives
recall = recall_score(y_test, model_predict)
print("Recall:", recall)

# F1 Score: Harmonic mean of precision and recall (balanced metric)
# Useful when classes are imbalanced
f1 = f1_score(y_test, model_predict)
print("F1 Score:", f1)
