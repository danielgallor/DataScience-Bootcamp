import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer.keys()
print(cancer['DESCR'])

df_feat = pd.DataFrame(cancer['data'], columns= cancer['feature_names'])
df_feat.head()
df_feat.info()
cancer['target']

df_target = pd.DataFrame(cancer['target'], columns= ['Cancer'])
df_target.head()

## Train Test Split

from sklearn.model_selection import train_test_split

X = df_feat
y = df_target

X_train, X_test, y_train, y_test = train_test_split( X, np.ravel(y), test_size=0.3, random_state=101)

from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

## Predicitons and Evaluations

predictions = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

## Gridsearch

from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10,100,1000], 'gamma': [1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(), param_grid,refit= True, verbose= 3)
grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_

grid_predictions = grid.predict(X_test)
print(classification_report(y_test, grid_predictions))
print(confusion_matrix(y_test, grid_predictions))


## EXCERCISE

from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url = url,width = 300, height = 300)
plt.show()

import seaborn as sns
iris = sns.load_dataset('iris')

#EDA
import pandas as pd 
import numpy as np

sns.pairplot(iris, hue = 'species')
plt.show()

setosa = iris[iris['species']=='setosa']
setosa
sns.kdeplot(setosa, y = 'sepal_length', x = 'sepal_width',cmap="plasma", shade=True, shade_lowest=False)
plt.show()

#Train Test
from sklearn.model_selection import train_test_split
X = iris.drop('species', axis = 1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))

#Gridsearch

from  sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1,10,100,1000], 'gamma': [1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(), param_grid,refit=True,verbose=3)

grid.fit(X_train, y_train)

grid_predict = grid.predict(X_test)

print(classification_report(y_test, grid_predict))
print(confusion_matrix(y_test, grid_predict))