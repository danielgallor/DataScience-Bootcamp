import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

yelp = pd.read_csv('C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\20-Natural-Language-Processing\\yelp.csv')
yelp.head()
yelp.info()
yelp.describe()
yelp.shape

yelp['text length'] = yelp['text'].apply(len)

#EDA
g = sns.FacetGrid(yelp, col='stars')
g = g.map(plt.hist, 'text length')
plt.show()

sns.boxplot(yelp, x = 'stars', y = 'text length', palette='rainbow')
plt.show()

sns.countplot(yelp, x='stars', palette='rainbow')
plt.show()

stars = yelp.groupby('stars').mean(numeric_only=True)
stars.head()

stars.corr()
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)
plt.show()

# NLP Classification