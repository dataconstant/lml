#implementing sigmoid function

import numpy as np

N=100
D=2

X=np.random.rand(N,D)
ones=np.ones(N)
ones=ones.reshape(100,1)
Xb=np.append(X,ones,axis=1)

print(Xb.shape)
print(ones.shape)

w=np.random.rand(D+1)
print(w.shape)

z=Xb.dot(w)
print(z.shape)

def sigmoid(z):
    return 1/(1+np.exp(-z))

#print (sigmoid(z))
print(sigmoid(z).shape)