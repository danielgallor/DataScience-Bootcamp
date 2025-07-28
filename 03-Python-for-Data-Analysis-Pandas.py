import numpy as np
import pandas as pd
from numpy.random import randn
import openpyxl

# np.random.seed(101)

# df= pd.DataFrame(randn(5,4), ['A','B','C','D','E'], ['W', 'X','Y','Z'])

# df['W']>0

# outside = ['G1','G1','G1','G2','G2','G2']
# inside = [1,2,3,1,2,3]
# hier_index = list(zip(outside,inside))
# hier_index = pd.MultiIndex.from_tuples(hier_index)

# df = pd.DataFrame(randn(6,2), hier_index,['A','B'])
# df.loc['G1'].loc[3]['A']

# df = pd.DataFrame({'A':[1,2,np.nan],
#                   'B':[5,np.nan,np.nan],
#                   'C':[1,2,3]})
# df.fillna(value= 'FILL VALUE')


# data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
#        'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
#        'Sales':[200,120,340,124,243,350]}

# df = pd.DataFrame(data)
# by_comp = df.groupby('Company').sum().loc['FB']
# by_comp


# df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
# df.head()

# df=pd.read_excel('C:\\Users\\Daniel.Gallo\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\03-Python-for-Data-Analysis-Pandas\\Excel_Sample.xlsx')
# df
# a=5

##### Excercise 1 ####

sal =  pd.read_csv('C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\04-Pandas-Exercises\\salaries.csv')
sal.head()
sal.info()
sal['BasePay'].mean()
sal['OvertimePay'].max()
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']
sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]
sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]
sal.groupby('Year')['BasePay'].mean()
sal['JobTitle'].nunique()
sal['JobTitle'].value_counts().head()
sum(sal[sal['Year']==2013]['JobTitle'].value_counts()==1)

sum(sal['JobTitle'].str.contains('Chief', case=False))

sal['title_len'] = sal['JobTitle'].apply(len)
sal['title_len']


##### Excercise 2 ####

ecom = pd.read_csv("C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\04-Pandas-Exercises\\Ecommerce Purchases.csv")
ecom.head()
ecom.info()
ecom['Purchase Price'].mean()
ecom['Purchase Price'].max()
ecom['Purchase Price'].min()
ecom[ecom['Language']=='en'].count()
ecom[ecom['Job']=='Lawyer'].count()
ecom['AM or PM'].value_counts()
ecom['Job'].value_counts().head(5)
ecom[ecom['Lot']=='90 WT']['Purchase Price']
ecom[ecom['Credit Card']== 4926535242672853]['Email']
ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price']>95)].count()
sum(ecom['CC Exp Date'].apply(lambda x:x[3:]) == '25')
ecom['Email'].apply(lambda x:x.split('@')[1]).value_counts().head()

