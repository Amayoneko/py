import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import numpy.linalg as la
rate=0.04
def f(img):
	U,s,V=la.svd(img)
	m,n=img.shape
	S=np.zeros((m,n))
	p=min(m,n)
	r=int(rate*p)
	S[:p,:p]=np.diag(s)
	return U[:,:r].dot(S[:r,:r]).dot(V[:r,:n])
img=mpimg.imread("2.jpg")
for j in range(3):
	img[...,j]=f(img[...,j]).copy()
plt.imshow(img)
plt.show()
