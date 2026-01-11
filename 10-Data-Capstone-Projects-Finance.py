from pandas_datareader import data, wb
import yfinance as yf
import requests, certifi
import pandas as pd
import numpy as np
import datetime
import pickle

df=pd.read_pickle("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\10-Data-Capstone-Projects\\all_banks.pkl")

session = requests.Session()
session.verify = certifi.where() 


start = datetime.datetime(2006,1,1)
end =  datetime.datetime(2016,1,1)

def get_stock_data(ticker, start, end, **kwargs):
    return yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=True, **kwargs)

C = get_stock_data("C", start, end)

