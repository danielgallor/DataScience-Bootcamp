
import pandas as pd
import numpy as np
import datetime
import pickle

bank_stocks=pd.read_pickle("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\10-Data-Capstone-Projects\\all_banks.pkl")

bank_stocks.xs(key= "Close", axis = 1, level = "Stock Info").max()

a=5
