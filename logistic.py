from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold='nan')
m = 1000
x_moons, y_moons = make_moons(m, noise=0.1, random_state=42)
"""
for i in range(m):
	if(y_moons[i]==0):
		plt.plot(x_moons[i,0],x_moons[i,1],'r.')
	else:
		plt.plot(x_moons[i,0],x_moons[i,1],'g.')
plt.show()
"""
A=np.c_[np.ones((m,1)),x_moons]
y=y_moons.reshape(-1,1)
import tensorflow as tf

