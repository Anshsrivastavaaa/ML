# ============================================================
# 1.py — Data Preprocessing & Exploratory Data Analysis (EDA)
#         on the Titanic "tested.csv" dataset
# ============================================================

# --- Import libraries ---
import pandas as pd              # For data manipulation and analysis (DataFrames)
import matplotlib.pyplot as pl   # For creating plots and visualizations
import numpy as np               # For numerical operations (arrays, math functions)
from sklearn.preprocessing import StandardScaler  # For feature scaling (z-score normalization)


# --- Load the dataset ---
# Read the Titanic test dataset from a CSV file into a pandas DataFrame
df = pd.read_csv("tested.csv")

# --- Step 1: Summary statistics ---
# Display statistical summary (count, mean, std, min, 25%, 50%, 75%, max) for numerical columns
df.describe()

# --- Step 2: Dataset info ---
# Display column names, data types, non-null counts, and memory usage
df.info()

# --- Step 3: First 10 rows ---
# Preview the first 10 rows to understand the structure of the data
df.head(10)

# --- Step 4: Last 10 rows ---
# Preview the last 10 rows to check the tail end of the dataset
df.tail(10)


# --- Handle missing values ---
# Count the number of null (missing) values in each column
df.isnull().sum()

# Fill missing 'Age' values with the mean age (avoids bias from dropping rows)
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill missing 'Fare' values with the mean fare
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# Fill missing 'Embarked' values with the mode (most frequent embarkation port)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())

# Drop any remaining rows that still have null values
df.dropna(inplace=True)

# --- Step 5: Verify no missing values remain ---
# Confirm that all null values have been handled
df.isnull().sum()

# --- Encode categorical variables ---
# Convert 'Sex' from text ('male'/'female') to numeric (0/1) for model compatibility
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' port names to numeric codes: S=0, C=1, Q=2
df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

# --- Step 6: Verify encoding ---
# Display last few rows to confirm the encoding was applied correctly
df.tail()


# --- Feature scaling ---
# Initialize StandardScaler to normalize features to mean=0, std=1
scaler = StandardScaler()

# Select the numerical column(s) to scale
numerical_cols = ['Fare']

# Fit the scaler on the 'Fare' column and transform it (z-score normalization)
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# --- Step 7: Verify scaling ---
# Display the first 10 rows to confirm 'Fare' has been standardized
df.head(10)


# --- Step 8: Visualization — Bar chart of Age by Sex ---
# Extract 'Sex' (0=male, 1=female) and 'Age' columns for plotting
x = df['Sex']
y = df['Age']

# Create a bar chart showing age values grouped by sex
pl.bar(x, y)

# Label the x-axis as 'Sex'
pl.xlabel('Sex')

# Label the y-axis as 'Age'
pl.ylabel('Age')

# Add a descriptive title to the chart
pl.title('Age by Sex (Individual Data Points)')

# Render and display the plot
pl.show()
