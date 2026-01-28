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

sns.displot((y_test-predictions), bins = 50)
plt.show()

from sklearn import metrics
metrics.mean_absolute_error(y_test, predictions)
metrics.mean_squared_error(y_test, predictions)
np.sqrt(metrics.mean_squared_error(y_test, predictions))


### PROJECT ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

customer = pd.read_csv("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\11-Linear-Regression\\Ecommerce Customers.csv")
customer.head()
customer.info()
customer.describe()

sns.jointplot(customer, x ="Time on Website", y = "Yearly Amount Spent")
plt.show()

sns.jointplot(customer, x ="Time on App", y = "Yearly Amount Spent")
plt.show()

sns.jointplot(customer, x = "Time on App", y = "Length of Membership", kind= "hex")
plt.show()

sns.pairplot(customer)
plt.show()

sns.lmplot(customer, y = "Yearly Amount Spent", x = "Length of Membership")
plt.show()

    # Train Model
from sklearn.model_selection import train_test_split

X = customer[[ 'Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = customer['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

print("Coefficient: \n", lm.coef_) 

   
    # Predict Test Data

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel("Y Test")
plt.ylabel("Predicted Y")
plt.show()


    # Evaluating the Model

from sklearn import metrics
MAE = metrics.mean_absolute_error(y_test, predictions)
MSE = metrics.mean_squared_error(y_test, predictions)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))

print("MAE:", MAE)
print("MSE:", MSE)
print("RMSE:", RMSE)

    # Residuals

sns.displot((y_test-predictions), bins = 50, kde = True)
plt.show()

coefficient = pd.DataFrame(lm.coef_, X.columns, columns=["Coefficient"])
coefficient
