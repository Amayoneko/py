import numpy as np
import tensorflow as tf
a=np.mat([[1,2,3],[4,5,6],[7,8,9]])
b=np.mat([[1],[2],[3]])
A=tf.placeholder(tf.int32,shape=(3,3))
B=tf.placeholder(tf.int32,shape=(3,1))
C=A+B
with tf.Session() as sess:
	print sess.run(C,feed_dict={A:a,B:b})
