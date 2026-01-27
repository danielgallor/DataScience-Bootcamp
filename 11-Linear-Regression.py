import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


USAhousing = pd.read_csv("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\11-Linear-Regression\\USA_Housing.csv")
USAhousing.head()

USAhousing.info()
USAhousing.describe()

sns.pairplot(USAhousing)
plt.show()

sns.displot(USAhousing["Price"], kde = True)
plt.show()

USAhousing.corr()
sns.heatmap(USAhousing.describe().corr(), annot =  True)
plt.show()
USAhousing.columns

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_, X.columns,columns=["Coefficient"])
coeff_df

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.show()