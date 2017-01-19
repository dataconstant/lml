import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loading data

df=pd.read_csv('data/data_1d.csv',header=-1,names=['x','y'])
x=df['x']
y=df['y']
x=np.array(x)
y=np.array(y)

plt.scatter(x,y)
plt.show()

den=x.dot(x)-x.mean()*x.sum()
a=(x.dot(y)-y.mean()*x.sum())/den
b=(y.mean()*x.dot(x)-x.mean()*x.dot(y))/den

# predicted y

ypred=a*x+b

# plotting results

plt.scatter(x,y)
plt.plot(x,ypred)
plt.show()

#checking prediction

d1 = y - ypred
d2 = y - y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)

