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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

# Prediction and Evaluation

