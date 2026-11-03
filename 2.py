import pandas as pd
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

#1
dataset = load_digits()
dataset.keys()

#2
dataset.data.shape

#3
dataset.data[0]

#4
dataset.data[0].reshape(8,8)

#5
from matplotlib import pyplot as plt
%matplotlib inline

plt.gray()
plt.matshow(dataset.data[0].reshape(8,8))

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

x = df
y = dataset.target

#6
y

#7
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled


#8
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=30)

model = LogisticRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test)

#9
x.shape

#10
pca = PCA(0.95)
x_pca = pca.fit_transform(x)
x_pca.shape

#11
pca.explained_variance_ratio_
pca.n_components_

x_train_pca, x_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=30)

#12
model = LogisticRegression(max_iter=1000)
model.fit(x_train_pca, y_train)
model.score(x_test_pca, y_test)

#13
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
x_pca.shape

#14
x_pca

#15
pca.explained_variance_ratio_

x = df
y = dataset.target
x_train_pca, x_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=30)

#16
model = LogisticRegression(max_iter=1000)
model.fit(x_train_pca, y_train)
model.score(x_test_pca, y_test)

#17
data = load_breast_cancer()
X = data.data
y = data.target
print("Original Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
start = time.time()

model_original = LogisticRegression(max_iter=1000)
model_original.fit(X_train_scaled, y_train)
prediction_original = model_original.predict(X_test_scaled)
end = time.time()

accuracy_original = accuracy_score(y_test, prediction_original)
time_original = end - start

#18
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)
print("Reduced Shape:", X_train_lda.shape)

start = time.time()
model_lda = LogisticRegression()
model_lda.fit(X_train_lda, y_train)
prediction_lda = model_lda.predict(X_test_lda)
end = time.time()
accuracy_lda = accuracy_score(y_test, prediction_lda)
time_lda = end - start

#19
print("\nComparison Results")

print("Original Features :", X_train_scaled.shape[1])
print("Reduced Features :", X_train_lda.shape[1])

print("Accuracy Before LDA :", accuracy_original)
print("Accuracy After LDA :", accuracy_lda)

print("Training Time Before LDA :", time_original)
print("Training Time After LDA :", time_lda)

print("\nClassification Report Without LDA")
print(classification_report(y_test, prediction_original))

print("\nClassification Report With LDA")
print(classification_report(y_test, prediction_lda))
