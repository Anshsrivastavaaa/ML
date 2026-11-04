# ============================================================
# 5.py — Logistic Regression with Regularization (L1 & L2)
#         Classifying laptop prices as high/low
# ============================================================

# --- Import libraries ---
import pandas as pd                                       # For data manipulation
import matplotlib.pyplot as plt                           # For creating plots
import seaborn as sns                                     # For statistical visualizations (heatmap)

from sklearn.model_selection import train_test_split      # For splitting data into train/test sets
from sklearn.preprocessing import StandardScaler          # For feature normalization (z-score)
from sklearn.linear_model import LogisticRegression       # Logistic Regression classifier
from sklearn.metrics import accuracy_score                # For computing classification accuracy

# --- Step 1: Load and preview dataset ---
# Read the laptop price dataset from CSV
df = pd.read_csv('laptop_price - dataset.csv')

# Display the first 5 rows to inspect the data
df.head(5)

# --- Step 2: Summary statistics ---
# Show count, mean, std, min, max, and quartiles for numerical columns
df.describe()

# --- Step 3: Dataset structure ---
# Display column names, data types, and non-null counts
df.info()

# --- Step 4: Check for missing values ---
# Count null values per column
df.isnull().sum()

# --- Step 5: Create binary target variable ---
# Calculate the average laptop price across the dataset
average_price = df["Price (Euro)"].mean()

# Create a new binary column: 0 = below average price, 1 = above average price
df["Price_Class"] = 0
df.loc[df["Price (Euro)"] > average_price, "Price_Class"] = 1

# Select numerical features for the model
x = df[["Inches", "CPU_Frequency (GHz)", "RAM (GB)", "Weight (kg)"]]

# Target variable: binary price class (0 = cheap, 1 = expensive)
y = df["Price_Class"]

# Display the class distribution to check for imbalance
print(df["Price_Class"].value_counts())


# --- Split into training and testing sets ---
# 80% train, 20% test; random_state=42 for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- Scale features ---
# Standardize features to mean=0, std=1 (important for Logistic Regression)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)      # Fit and transform on training data
x_test = scaler.transform(x_test)            # Transform test data using training statistics

# --- Step 6: Train baseline Logistic Regression ---
# Train a standard Logistic Regression model (default L2 penalty)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Predict on test set
model_pred = model.predict(x_test)

# Calculate and display accuracy
logistic_accuracy = accuracy_score(y_test, model_pred)
print("Logistic Regression Accuracy:", logistic_accuracy)

# --- Step 7: Feature correlation analysis ---
# Compute pairwise Pearson correlation between features
correlation = x.corr()
print("Correlation Matrix:")
print(correlation)

# Visualize the correlation matrix as a heatmap
# 'annot=True' displays correlation values; 'coolwarm' uses blue-to-red color scale
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# --- Step 8: Ridge Regression (L2 regularization) ---
# L2 penalty shrinks all coefficients towards zero (but doesn't eliminate any)
# Helps prevent overfitting by penalizing large coefficient values
ridge = LogisticRegression(penalty="l2", max_iter=1000)
ridge.fit(x_train, y_train)
ridge_pred = ridge.predict(x_test)

ridge_accuracy = accuracy_score(y_test, ridge_pred)
print("Ridge Accuracy:", ridge_accuracy)

# --- Step 9: Lasso Regression (L1 regularization) ---
# L1 penalty can drive some coefficients to exactly zero (feature selection)
# 'liblinear' solver is required for L1 penalty in LogisticRegression
lasso = LogisticRegression( penalty="l1", solver="liblinear", max_iter=1000)
lasso.fit(x_train, y_train)
lasso_pred = lasso.predict(x_test)

lasso_accuracy = accuracy_score(y_test, lasso_pred)
print("Lasso Accuracy:", lasso_accuracy)

# --- Step 10: Compare all three models visually ---
# Create a bar chart comparing accuracies of Logistic, Ridge, and Lasso
plt.bar( ["Logistic", "Ridge", "Lasso"], [logistic_accuracy, ridge_accuracy, lasso_accuracy], color = ['blue','red', 'green'])

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")

plt.show()
