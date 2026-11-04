# ============================================================
# 4.py — Polynomial Regression
#         Fitting curves of different degrees to Salary data
# ============================================================

# --- Import libraries ---
import pandas as pd                                       # For reading CSV and data manipulation
import matplotlib.pyplot as plt                           # For creating plots
from sklearn.model_selection import train_test_split      # For train/test splitting
from sklearn.preprocessing import PolynomialFeatures      # For generating polynomial features (x², x³, etc.)
from sklearn.linear_model import LinearRegression         # Linear Regression (applied to polynomial features)
from sklearn.metrics import mean_squared_error            # For computing Mean Squared Error


# --- Load the dataset ---
# Read the Salary dataset — contains 'YearsExperience' and 'Salary' columns
df = pd.read_csv("/content/Salary.csv")

# --- Step 1: Preview the data ---
# Display the first 5 rows to understand the dataset structure
df.head()

# --- Prepare features and target ---
# Independent variable: years of work experience
x = df['YearsExperience']

# Dependent variable: salary (what we want to predict)
y = df['Salary']

# --- Split into training and testing sets ---
# 80% train, 20% test; random_state=42 for reproducibility
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)


# --- Step 2: Train and evaluate polynomial models of degrees 1–4 ---
# Test polynomial degrees: 1 (linear), 2 (quadratic), 3 (cubic), 4 (quartic)
degrees = [1, 2, 3, 4]

for degree in degrees:

    # Create polynomial feature transformer for the current degree
    # e.g., degree=2 transforms [x] into [1, x, x²]
    poly = PolynomialFeatures(degree=degree)

    # Fit on training data and transform: reshape(-1, 1) converts 1D array to 2D column
    x_train_poly = poly.fit_transform(x_train.values.reshape(-1, 1))

    # Transform test data using the same polynomial mapping (no re-fitting)
    x_test_poly = poly.transform(x_test.values.reshape(-1, 1))

    # Create and train a Linear Regression model on the polynomial features
    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    # Predict salary on the test set
    y_pred = model.predict(x_test_poly)

    # Calculate Mean Squared Error to measure prediction quality
    error = mean_squared_error(y_test, y_pred)

    # Print the degree and its corresponding MSE
    print("Degree:", degree)
    print("MSE:", error)
    print()

# --- Step 3: Visualize all polynomial fits ---
# Scatter plot of the actual data points (experience vs salary)
plt.scatter(x, y)

# Fit and plot a polynomial curve for each degree on the full dataset
for degree in [1, 2, 3, 4]:

    # Generate polynomial features for the current degree
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x.values.reshape(-1, 1))

    # Train a Linear Regression on the full dataset (for visualization only)
    model = LinearRegression()
    model.fit(x_poly, y)

    # Predict using the trained model to get the fitted curve
    y_pred = model.predict(poly.transform(x.values.reshape(-1, 1)))

    # Plot the fitted curve with a label indicating its degree
    plt.plot(x, y_pred, label="Degree " + str(degree))

# Add axis labels, title, and legend
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Polynomial Regression with Different Degrees")
plt.legend()

# Display the plot
plt.show()
