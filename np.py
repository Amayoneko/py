import numpy as np
A=np.random.rand(4,4)
B=np.random.rand(4,4)
print np.sort(np.linalg.eigvals(A.dot(B)))
print np.sort(np.linalg.eigvals(B.dot(A)))
