# ============================================================
# 3.py — Multiple Linear Regression
#         Predicting house prices from multiple features
# ============================================================

# --- Import libraries ---
import pandas as pd                                      # For data manipulation with DataFrames
import numpy as np                                       # For numerical operations (e.g., sqrt)
from sklearn.model_selection import train_test_split     # For splitting data into train/test sets
from sklearn.metrics import mean_squared_error, r2_score # For evaluating regression performance
from sklearn.linear_model import LinearRegression        # Multiple Linear Regression model

# --- Step 1: Load and preview the dataset ---
# Read the housing dataset from CSV into a DataFrame
df = pd.read_csv("Housing.csv")

# Display the first 5 rows to understand the data structure and column names
df.head(5)

# --- Step 2: Dataset info ---
# Display column data types, non-null counts, and memory usage
df.info()

# --- Step 3: Summary statistics ---
# Show statistical summary (mean, std, min, max, quartiles) for all numerical columns
df.describe()

# --- Step 4: Check for missing values ---
# Count null values per column to decide if imputation or dropping is needed
df.isnull().sum()

# --- Prepare features and target ---
# Select 5 numerical features as independent variables (predictors)
x = df[['area', 'bedrooms', 'bathrooms', 'stories','parking']]

# Select 'price' as the dependent variable (what we want to predict)
y = df[['price']]

# --- Split into training and testing sets ---
# 80% for training, 20% for testing; random_state=42 ensures reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- Step 5: Train the Linear Regression model ---
# Create a Linear Regression model instance
model = LinearRegression()

# Fit (train) the model on the training data — learns coefficients for each feature
model.fit(x_train, y_train)

# Use the trained model to predict prices on the unseen test data
y_pred = model.predict(x_test)

# --- Step 6: Evaluate model performance ---
# Mean Squared Error: average of squared differences between actual and predicted values
# Lower MSE = better model
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error: square root of MSE, in the same units as the target variable
# Easier to interpret than MSE (e.g., error in dollars for price)
rmse = np.sqrt(mse)

# R-squared: proportion of variance in the target explained by the model
# Ranges from 0 to 1; closer to 1 means better fit
r2 = r2_score(y_test, y_pred)

# Print all evaluation metrics
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
