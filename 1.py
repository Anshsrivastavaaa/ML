import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("tested.csv")

#1
df.describe()

#2
df.info()

#3
df.head(10)

#4
df.tail(10)


df.isnull().sum()
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())

df.dropna(inplace=True)

#5
df.isnull().sum()

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

#6
df.tail()



scaler = StandardScaler()
numerical_cols = ['Fare']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

#7
df.head(10)


#8
x = df['Sex']
y = df['Age']
pl.bar(x, y)
pl.xlabel('Sex')
pl.ylabel('Age')
pl.title('Age by Sex (Individual Data Points)')
pl.show()
