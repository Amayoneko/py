import numpy as np
from numpy import linalg as la
A=np.mat([[-1,0,2],[1,2,-1]])
U,sg,VT=la.svd(A)
#print P.dot(np.diag(dg)).dot(la.inv(P))
"""print U
t=U[:,0].copy();
U[:,0]=U[:,1].copy();
print U
U[:,1]=t.copy()
print U"""
p,q=la.eig(A.T.dot(A))
print p
print q
print U
print sg
print VT.T
