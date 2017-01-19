import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot

#Loading Data

df=pd.read_csv('data/data_2d.csv',header=-1,names=['a','b','c'])

X1=df['a']
X2=df['b']
Y=df['c']
X1=np.array(X1)
X2=np.array(X2)
Y=np.array(Y)

#Adding bias

bias=np.ones(100,)
X=np.vstack([X1,X2,bias])
#X=np.vstack([X])
X=np.transpose(X)

# Two dimensional linear regression and predicting y

w=np.linalg.solve(np.dot(X.T,X), np.dot(X.T,Y))
Yp=np.dot(X,w)

#Checking y

d1 = Y - Yp
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print ('the r-squared is:',r2)
