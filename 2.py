# ============================================================
# 2.py — Dimensionality Reduction with PCA and LDA
#         Comparing model performance before and after reduction
# ============================================================

# --- Import libraries ---
import pandas as pd    # For data manipulation using DataFrames
import time            # For measuring training time

from sklearn.linear_model import LogisticRegression        # Logistic Regression classifier
from sklearn.model_selection import train_test_split       # For splitting data into train/test sets
from sklearn.preprocessing import StandardScaler           # For z-score feature normalization
from sklearn.datasets import load_digits                   # Handwritten digits dataset (8x8 images)

from sklearn.decomposition import PCA                      # Principal Component Analysis for dimensionality reduction
from sklearn.datasets import load_breast_cancer             # Breast cancer classification dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA for supervised dimensionality reduction
from sklearn.metrics import accuracy_score, classification_report     # For evaluating model performance

# ========================
# PART 1: PCA on Digits Dataset
# ========================

# --- Step 1: Load the digits dataset ---
# Load the handwritten digits dataset (1797 samples, 64 features each — 8x8 pixel images)
dataset = load_digits()
dataset.keys()  # Display available keys: 'data', 'target', 'feature_names', etc.

# --- Step 2: Check data dimensions ---
# Show the shape of the data array (1797 samples × 64 features)
dataset.data.shape

# --- Step 3: Inspect a single sample ---
# Display the raw pixel values of the first digit image (64-element flat array)
dataset.data[0]

# --- Step 4: Reshape to 8×8 grid ---
# Reshape the flat array into its original 8×8 pixel grid to visualize the digit
dataset.data[0].reshape(8,8)

# --- Step 5: Visualize the digit image ---
from matplotlib import pyplot as plt
%matplotlib inline  # Jupyter magic to display plots inline

# Set the colormap to grayscale for the digit image
plt.gray()

# Display the first digit as an 8×8 grayscale image
plt.matshow(dataset.data[0].reshape(8,8))

# --- Prepare features and target ---
# Create a DataFrame from the dataset with proper column names
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# Assign features (all 64 pixel columns) and target (digit labels 0–9)
x = df
y = dataset.target

# --- Step 6: Inspect target labels ---
# Display the target array (digit labels: 0, 1, 2, ..., 9)
y

# --- Step 7: Standardize features ---
# Scale features to mean=0, std=1 (important for PCA and Logistic Regression)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled  # Display the scaled data

# --- Step 8: Baseline model (no PCA) ---
# Split scaled data into 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=30)

# Train a Logistic Regression model on all 64 features
model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluate accuracy on the test set (baseline accuracy with all features)
model.score(x_test, y_test)

# --- Step 9: Check original feature count ---
# Display the shape — should be (1797, 64), i.e., 64 features
x.shape

# --- Step 10: Apply PCA — retain 95% of variance ---
# PCA reduces dimensions while keeping 95% of the total variance
pca = PCA(0.95)
x_pca = pca.fit_transform(x)

# Show the reduced shape — fewer features now (e.g., ~28 components)
x_pca.shape

# --- Step 11: Inspect PCA results ---
# Show what fraction of variance each principal component explains
pca.explained_variance_ratio_

# Show how many components were needed to capture 95% variance
pca.n_components_

# Split the PCA-reduced data into train/test
x_train_pca, x_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=30)

# --- Step 12: Train model on PCA-reduced data (95% variance) ---
# Train Logistic Regression on the reduced feature set
model = LogisticRegression(max_iter=1000)
model.fit(x_train_pca, y_train)

# Evaluate — compare this accuracy with the baseline (Step 8)
model.score(x_test_pca, y_test)

# --- Step 13: Aggressive PCA — reduce to just 2 components ---
# Reduce to only 2 dimensions (useful for 2D visualization)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

# Shape should be (1797, 2)
x_pca.shape

# --- Step 14: Inspect 2-component data ---
# Display the 2D projected data
x_pca

# --- Step 15: Variance explained by 2 components ---
# Show how much variance these 2 components capture (much less than 95%)
pca.explained_variance_ratio_

# --- Step 16: Train model on 2-component PCA data ---
# Re-assign features and target
x = df
y = dataset.target

# Split the 2-component PCA data
x_train_pca, x_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=30)

# Train and evaluate — accuracy will be lower due to heavy dimensionality reduction
model = LogisticRegression(max_iter=1000)
model.fit(x_train_pca, y_train)
model.score(x_test_pca, y_test)

# ========================
# PART 2: LDA on Breast Cancer Dataset
# ========================

# --- Step 17: Load and prepare breast cancer data ---
# Load the breast cancer dataset (569 samples, 30 features, binary target)
data = load_breast_cancer()
X = data.data
y = data.target
print("Original Shape:", X.shape)  # (569, 30)

# Split into train/test sets with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features to zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler (no re-fitting on test data!)

# Train baseline Logistic Regression on all 30 original features
start = time.time()
model_original = LogisticRegression(max_iter=1000)
model_original.fit(X_train_scaled, y_train)
prediction_original = model_original.predict(X_test_scaled)
end = time.time()

# Calculate baseline accuracy and training time
accuracy_original = accuracy_score(y_test, prediction_original)
time_original = end - start

# --- Step 18: Apply LDA for dimensionality reduction ---
# LDA reduces to at most (n_classes - 1) dimensions; for binary = 1 component
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)  # LDA is supervised — needs y
X_test_lda = lda.transform(X_test_scaled)
print("Reduced Shape:", X_train_lda.shape)  # (455, 1) — 30 features → 1 feature

# Train Logistic Regression on the single LDA component
start = time.time()
model_lda = LogisticRegression()
model_lda.fit(X_train_lda, y_train)
prediction_lda = model_lda.predict(X_test_lda)
end = time.time()

# Calculate LDA accuracy and training time
accuracy_lda = accuracy_score(y_test, prediction_lda)
time_lda = end - start

# --- Step 19: Print comparison results ---
# Compare original (30 features) vs LDA-reduced (1 feature) performance
print("\nComparison Results")

print("Original Features :", X_train_scaled.shape[1])  # 30
print("Reduced Features :", X_train_lda.shape[1])      # 1

print("Accuracy Before LDA :", accuracy_original)
print("Accuracy After LDA :", accuracy_lda)

print("Training Time Before LDA :", time_original)
print("Training Time After LDA :", time_lda)

# Detailed precision/recall/f1 report for the original model
print("\nClassification Report Without LDA")
print(classification_report(y_test, prediction_original))

# Detailed precision/recall/f1 report for the LDA-reduced model
print("\nClassification Report With LDA")
print(classification_report(y_test, prediction_lda))
