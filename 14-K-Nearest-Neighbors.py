import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\14-K-Nearest-Neighbors\\Classified Data.csv")
df.head()

## Standardise Variables

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop(['TARGET CLASS','Unnamed: 0'], axis = 1))

scaled_features = scaler.transform(df.drop(['TARGET CLASS','Unnamed: 0'], axis =1))

df_feat = pd.DataFrame(scaled_features, columns = df.columns[1:-1])

## Train/Test Split

from sklearn.model_selection import train_test_split
X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

## KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

## Prediction and Evaluation

prediction = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

## Choosing a K Value

error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors= i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color = 'blue', linestyle = 'dashed', marker ='o', markerfacecolor = 'red', markersize = 10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

## Comparisson to higher K value

knn = KNeighborsClassifier(n_neighbors= 17)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

### EXCERCISE 

