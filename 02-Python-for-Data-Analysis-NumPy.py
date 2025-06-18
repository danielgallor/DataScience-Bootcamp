import numpy as np
#Q1
np.zeros(10)
#Q2
np.ones(10)
#Q3
np.ones(10)*5
#Q4
np.arange(10,51)
#Q5
np.arange(10,50,2)
#Q6
arr=np.arange(0,9)
arr.reshape(3,3)
#Q7
np.eye(3)
#Q8
np.random.rand(1)
#Q9
np.random.randn(25)
#Q10
np.arange(0.01,1.01,0.01).reshape(10,10)
#Q11
np.linspace(0,1,20)
#Q12
mat = np.arange(1,26).reshape(5,5)
mat
mat[2:,1:]
#Q13
mat
mat[3,4]
#Q14    
mat[:3,1]
#Q15
mat[4]
#Q16
mat[3:]
#Q17
mat.sum()
#Q18
mat.std()
#Q19
mat.sum(axis=0)

arr2d = np.arange(50).reshape(5,10)
arr2d[3,3:6]