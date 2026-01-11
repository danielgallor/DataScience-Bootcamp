import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\10-Data-Capstone-Projects\\911.csv')
df.info()

df.head(3)
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df['title'].nunique()


df['reason'] = df['title'].apply(lambda title: title.split(':')[0])
df['reason'].value_counts()

sns.countplot(x = 'reason', data = df, palette = 'viridis')
plt.show()

type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day'] = df['timeStamp'].apply(lambda time: time.dayofweek)

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day'].map(dmap)

sns.countplot(x = 'Day of Week', data = df, hue = 'reason')
plt.show()

sns.countplot(x = 'Month', data = df, hue = 'reason')
plt.show()


byMonth = df.groupby('Month').count()
byMonth

byMonth['twp'].plot()
plt.show()

sns.lmplot(x ='Month', y = 'twp', data = byMonth.reset_index())
plt.show()

df['Date'] = df['timeStamp'].apply(lambda t:t.date())
df.groupby('Date').count()['twp'].plot()
plt.show()

df[df['reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.show()

dayHour = df.groupby(by =[ 'Day of Week', 'Hour']).count()['reason'].unstack()
dayHour.head()

sns.heatmap(dayHour)
plt.show()

sns.clustermap(dayHour)
plt.show()


dayMonth = df.groupby(by =[ 'Day of Week', 'Month']).count()['reason'].unstack()
dayMonth.head()

sns.heatmap(dayMonth)
plt.show()

sns.clustermap(dayMonth)
plt.show()


a = 5

