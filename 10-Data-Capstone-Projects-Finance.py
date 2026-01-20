import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
import seaborn as sns

bank_stocks=pd.read_pickle("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\10-Data-Capstone-Projects\\all_banks.pkl")
bank_stocks.head(5)

tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

bank_stocks.xs(key= "Close", axis = 1, level = "Stock Info").max()

returns = pd.DataFrame()

for tick in tickers:
    returns[tick+ ' Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()

sns.pairplot(returns[1:])
plt.show()

returns.idxmax()
returns.idxmin()

returns.std()





a=5
