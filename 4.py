import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


df = pd.read_csv("/content/Salary.csv")

#1
df.head()

x = df['YearsExperience']
y = df['Salary']

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)


#2
degrees = [1, 2, 3, 4]

for degree in degrees:

    poly = PolynomialFeatures(degree=degree)

    x_train_poly = poly.fit_transform(x_train.values.reshape(-1, 1))
    x_test_poly = poly.transform(x_test.values.reshape(-1, 1))

    model = LinearRegression()

    model.fit(x_train_poly, y_train)

    y_pred = model.predict(x_test_poly)

    error = mean_squared_error(y_test, y_pred)

    print("Degree:", degree)
    print("MSE:", error)
    print()

#3
plt.scatter(x, y)

for degree in [1, 2, 3, 4]:

    poly = PolynomialFeatures(degree=degree)

    x_poly = poly.fit_transform(x.values.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_poly, y)

    y_pred = model.predict(poly.transform(x.values.reshape(-1, 1)))

    plt.plot(x, y_pred, label="Degree " + str(degree))

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Polynomial Regression with Different Degrees")
plt.legend()
plt.show()
