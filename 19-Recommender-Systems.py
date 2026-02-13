import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


colums_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\19-Recommender-Systems\\u.data.csv', sep = '\t', names= colums_names)
df.head()

movie_titles = pd.read_csv('C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\19-Recommender-Systems\\Movie_Id_Titles.csv' \
'')
movie_titles.head()

df = pd.merge(df, movie_titles, on = 'item_id')
df.head()

## EDA
sns.set_style('white')

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

df.groupby('title')['rating'].count().sort_values(ascending=False).head()

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = df.groupby('title')['rating'].count()
ratings.head()

plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)
plt.show()

plt.figure(figsize=(10,4))
ratings['rating'].hist(bins = 70)
plt.show()

sns.jointplot(ratings, x= 'rating', y = 'num of ratings')
plt.show()

## Recommending Similar Movies

moviemat = df.pivot_table(index = 'user_id', columns = 'title', values = 'rating')
moviemat.head()

ratings.sort_values('num of ratings', ascending= False).head()

starwars_user_ratings = moviemat['Star Wars (1977)']
lailair_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()


#Correlating other movies based on starwars/liarliar ratings
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(lailair_user_ratings)

#data frame of results drop na
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace= True)
corr_starwars.head()

# filtering ratings based on more than 100 people watched the movie
corr_starwars.sort_values('Correlation', ascending=False).head(10)
corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending=False).head()

#Sam with  LiarLiar
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace= True)
corr_liarliar.head()


corr_liarliar.sort_values('Correlation', ascending=False).head(10)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar.head()
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation', ascending=False).head()

