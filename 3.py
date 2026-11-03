import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

#1
df = pd.read_csv("Housing.csv")
df.head(5)

#2
df.info()

#3
df.describe()

#4
df.isnull().sum()

x = df[['area', 'bedrooms', 'bathrooms', 'stories','parking']]
y = df[['price']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#5
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

#6
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
