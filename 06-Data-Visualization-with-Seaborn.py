import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

#--------- Distribution Plots --------
tips = sns.load_dataset('tips')
tips.head()
sns.distplot(tips['total_bill'], bins=30) # bins = intervals that divide the entire range of data into segments



sns.jointplot(x='total_bill',y='tip',data=tips, kind='scatter') # match two displots together. kind is the type of plot you want  hex, reg, 

sns.pairplot(tips,hue='sex', palette='coolwarm') # creates a grid of plot with every possible numerical combination of the data frame

sns.rugplot(tips['total_bill']) #draw a dash on everypoint in the distribution

# Kdeplots - Kernel Density Estimation plots. 
# Don't worry about understanding this code!
# It's just for the diagram below
#Create dataset
dataset = np.random.randn(25)
dataset

# Create another rugplot
sns.rugplot(dataset)

# Set up the x-axis for the plot
x_min = dataset.min() - 2
x_max = dataset.max() + 2

# 100 equally spaced points from x_min to x_max
x_axis = np.linspace(x_min,x_max,100)

# Set up the bandwidth, for info on this:
url = 'http://en.wikipedia.org/wiki/Kernel_density_estimation#Practical_estimation_of_the_bandwidth'

bandwidth = ((4*dataset.std()**5)/(3*len(dataset)))**.2


# Create an empty kernel list
kernel_list = []

# Plot each basis function
for data_point in dataset:
    
    # Create a kernel for each point and append to list
    kernel = stats.norm(data_point,bandwidth).pdf(x_axis)
    kernel_list.append(kernel)
    
    #Scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * .4
    plt.plot(x_axis,kernel,color = 'grey',alpha=0.5)

plt.ylim(0,1)

# To get the kde plot we can sum these basis functions.

# Plot the sum of the basis function
sum_of_kde = np.sum(kernel_list,axis=0)

# Plot figure
fig = plt.plot(x_axis,sum_of_kde,color='indianred')

# Add the initial rugplot
sns.rugplot(dataset,c = 'indianred')

# Get rid of y-tick marks
plt.yticks([])

# Set title
plt.suptitle("Sum of the Basis Functions")
sns.kdeplot(tips['total_bill'])
plt.show()

#------------- Categorical Plots --------

tips = sns.load_dataset('tips')
tips.head()

sns.barplot(x='sex',y='total_bill',data=tips)
sns.countplot(x='sex', data=tips)
sns.boxplot(x='day', y='total_bill',hue="smoker",data=tips, palette="coolwarm")
sns.violinplot(x="day", y="total_bill", data=tips,hue='sex',split=True,palette='Set1')
sns.stripplot(x="day", y="total_bill", data=tips,jitter=True,hue='sex',palette='Set1')

plt.show()


#------------- Matrix Plots --------
# Heatmap

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
tips.head()
tc = tips.corr(numeric_only=True)
sns.heatmap(tc)

fp = flights.pivot_table(index='month', columns ='year', values = 'passengers')

sns.heatmap(fp,cmap = 'coolwarm', linecolor ='white', linewidths = 1)

plt.show()

# Clustermap

sns.clustermap(fp, standard_scale=1)

plt.show()

#------------- Grids ------------
#Pairgrid
iris = sns.load_dataset('iris')
iris.head()

g = sns.PairGrid(iris)

g.map_diag(sns.histplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)

plt.show()

#Pairplot
sns.pairplot(iris)

plt.show()

#FacetGrid

tips = sns.load_dataset('tips')
tips.head()
g = sns.FacetGrid(data=tips, col="time", row= "smoker")
g = g.map(plt.hist, "total_bill")

plt.show()

#---------- Regression Plots ------
tips = sns.load_dataset('tips')
tips.head()
sns.lmplot(x='total_bill',y='tip',data=tips, hue='sex')
sns.lmplot(x='total_bill',y='tip',data=tips, col='sex')

plt.show()


#                                                          -- EXERCISES --
sns.set_style('whitegrid')
titanic = sns.load_dataset('titanic')
titanic.head()

sns.jointplot(x= 'fare', y='age', data=titanic)

sns.histplot(titanic['fare'],bins=30, color = 'red')

sns.boxplot(x='class', y= 'age', data=titanic, palette='rainbow')

sns.swarmplot(x='class', y= 'age', data=titanic, palette='rainbow')

sns.countplot(x='sex', data=titanic)

sns.heatmap(titanic.corr(numeric_only=True))

g = sns.FacetGrid(titanic, col='sex')
g= g.map(plt.hist, 'age')
plt.show()