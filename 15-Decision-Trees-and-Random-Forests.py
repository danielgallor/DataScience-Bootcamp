import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\15-Decision-Trees-and-Random-Forests\\kyphosis.csv")
df.head()

# EDA

sns.pairplot(df, hue = 'Kyphosis', palette= 'Set1')
plt.show()

# Train/Test Split

from sklearn.model_selection import train_test_split

X = df.drop('Kyphosis', axis = 1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

# Prediction and Evaluation

predict = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predict))
print(confusion_matrix(y_test, predict))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators= 200)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print(classification_report(y_test, rfc_pred))
print(confusion_matrix(y_test, rfc_pred))