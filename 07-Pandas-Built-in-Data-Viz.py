import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\07-Pandas-Built-in-Data-Viz\\df1.csv', index_col = 0)
df1.head()
df2 = pd.read_csv('C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\07-Pandas-Built-in-Data-Viz\\df2.csv')
df2.head()

df1['A'].hist(bins=30)
df1['A'].plot(kind='hist')
df1['A'].plot.hist()

df2.plot.area(alpha = 0.4)
df2.plot.bar(stacked=True)

df1.plot.line(y='B', figsize=(12,3),lw=1)

df1.plot.scatter(x='A', y='B', c='C', cmap='coolwarm')
df1.plot.scatter(x='A', y='B', s=df1['C']*100)

df2.plot.box() # if want speficic colums df2[['a','b']]

df = pd.DataFrame(np.random.randn(1000,2), columns = ['a','b'])
df.head()
df.plot.hexbin(x='a', y='b', gridsize =25)

df2.plot.density()

plt.show()


#------ Exercises ------

df3 =pd.read_csv('C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\07-Pandas-Built-in-Data-Viz\\df3.csv')
df3.info()

df3.plot.scatter(x='a', y='b', c='red',s=50, figsize=(12,3))
df3['a'].plot.hist()
plt.style.use('ggplot')
df3['a'].plot.hist(edgecolor='white', alpha=0.5,bins=25)
df3[['a','b']].plot.box()
df3['d'].plot.density(lw= 5, linestyle='dashed')
df3.loc[0,31].plot.area()
df3.iloc[0:31].plot.area()

plt.show()