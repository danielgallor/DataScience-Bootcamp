import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\13-Logistic-Regression\\titanic_train.csv")
train.head()
train.info()
train.describe()

## Explore Data

sns.heatmap(train.isnull(),yticklabels = False,cbar = False, cmap = "viridis")
plt.show()

sns.set_style("whitegrid")
sns.countplot(train,x = 'Survived',palette = 'RdBu_r')
plt.show()


sns.countplot(train,x = 'Survived',palette = 'RdBu_r', hue = 'Sex')
plt.show()

sns.countplot(train,x = 'Survived', hue = 'Pclass')
plt.show()


sns.displot(train['Age'].dropna(), bins = 30, color = 'darkred')
plt.show()

sns.countplot(train, x = 'SibSp')
plt.show()

sns.displot(train['Fare'], bins = 40)
plt.show()


## Clean Data

plt.figure(figsize= (12, 7))
sns.boxplot(train, x = 'Pclass', y = 'Age', palette= 'winter')
plt.show()


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
        
    else:
        return Age
    
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis = 1)

sns.heatmap(train.isnull(),yticklabels = False,cbar = False, cmap = "viridis")
plt.show()

train.drop('Cabin',axis = 1, inplace = True)                    
train.head()

train.dropna(inplace = True)
sns.heatmap(train.isnull(),yticklabels = False,cbar = False, cmap = "viridis")
plt.show()

## Categorical Features

sex = pd.get_dummies(train['Sex'], drop_first = True, dtype = int)
embark = pd.get_dummies(train['Embarked'], drop_first = True, dtype = int)

train.drop(['PassengerId','Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)
train = pd.concat([train, sex, embark], axis = 1)
train.head()

## Logistical Regression Model
    # Train/Test Split

X = train.drop('Survived', axis = 1)
y = train['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

## Evaluate Model 

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))


