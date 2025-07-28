import matplotlib.pyplot as plt
import numpy as np

plt.show()  # This will display an empty plot window
x = np.linspace(0,5,11)
y = x**2

#FUNCTIONAL METHOD
plt.plot(x,y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')

plt.subplot(1,2,1)
plt.plot(x,y,'r')

plt.subplot(1,2,2)
plt.plot(y,x,'b')


plt.show()

#OBJECT ORIENTED METHOD - MORE USEFUL
fig =  plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8])  # [left, bottom, width, height]
plt.show()

fig = plt.figure()   # Method to create various plots the same blank canvas modifying components for each in order
axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([0.2,0.2,0.6,0.6])

axes1.plot(x,y)
axes1.set_title('LARGER PLOT')
axes2.plot(y,x)
axes2.set_title('SMALLER PLOT')
plt.show()

fig,axes = plt.subplots(1,2)  # Method to create various plots in the same blank canvas at the same time
axes[0].plot(x,y)
axes[1].plot(y,x)    # sub plot allows to create figures as a list
plt.show()


fig =  plt.figure(figsize=(8,2)) #figsize is a tuple of the width and height of the figure in inches
ax = fig.add_axes([0,0,1,1]) 
ax.plot(x,y)
plt.show()

fig =  plt.subplots(2,1,figsize=(8,2)) #figsize is a tuple of the width and height of the figure in inches
ax = fig.add_axes([0,0,1,1]) 
ax[0].plot(x,y)
ax[1].plot(y,x)
plt.tight_layout() # fix if plots overlap - use it always as precaution
plt.show()

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(x,x**2,label='x**2')
ax.plot(x,x**3, label='x**3')

plt.tight_layout() # fix if plots overlap - use it always as precaution
ax.legend(loc=0)   # SHOW LEGENDS, LOCATION 0 is best
plt.show()

### Excercises ###

x = np.arange(0,100)
y = x*2
z = x**2

#1
fig = plt.figure()
ax= fig.add_axes([0,0,1,1])
ax.plot(x,y)

#2
fig = plt.figure()
ax1= fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,0.2,0.2])
ax1.plot(x,y)
ax2.plot(x,y)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

#3
fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5,0.4,0.4])
ax1.plot(x,z)  
ax2.plot(x,y)
ax2.set_xlim([20,22])       # Axis Range Limits
ax2.set_ylim([30,50])       # Axis Range Limits
ax2.set_title('Zoom')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

#4
fig,axes = plt.subplots(1,2)
axes[0].plot(x,y,'b')
axes[1].plot(x,z,'r')



plt.show()