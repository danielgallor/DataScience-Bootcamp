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

## EXCERCISE

loans = pd.read_csv("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\15-Decision-Trees-and-Random-Forests\\loan_data.csv")
loans.head()
loans.info()
loans.describe()

# EDA

sns.displot(loans, x = 'fico', hue = 'credit.policy', bins = 30)
plt.show()

sns.displot(loans, x = 'fico', hue = 'not.fully.paid', bins = 30)
plt.show()

sns.countplot(loans, x = 'purpose', hue = 'not.fully.paid')
plt.show()

sns.jointplot(loans, x = 'fico', y = 'int.rate')
plt.show()

sns.lmplot(loans, x = 'fico', y = 'int.rate', col ='not.fully.paid', hue = 'credit.policy', palette= 'Set1' )
plt.show()

# Setting Up Data

cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns = cat_feats, drop_first=True)
 
# Train Test

from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid', axis =1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 101)

# Train Decision Tree Model

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

prediction = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))


# Train Random Forest Model

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfc_pred))

print(classification_report(y_test, rfc_pred))
