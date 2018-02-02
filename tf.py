import numpy as np
import tensorflow as tf
a=np.mat([[1,2,3],[1,1,1]],dtype=np.float32)
A=tf.nn.softmax(a)
sess=tf.InteractiveSession()
print sess.run(A)
