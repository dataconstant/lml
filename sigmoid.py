#implementing sigmoid function

import numpy as np

N=100
D=2

X=np.random.rand(N,D)
ones=np.ones(N)
ones=ones.reshape(100,1)
Xb=np.append(X,ones,axis=1)

print("Xb Shape - ",Xb.shape)
print("ones Shape - ",ones.shape)

w=np.random.rand(D+1)
print("w Shape - ",w.shape)

z=Xb.dot(w)
print("z Shape - ",z.shape)

def sigmoid(z):
    return 1/(1+np.exp(-z))

print(sigmoid(z))
print("sigmoid Shape - ",sigmoid(z).shape)