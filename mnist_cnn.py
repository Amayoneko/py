import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
def weight_variable(shape):
	initial=tf.truncated_normal(shape=shape,stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)
def conv2d(X,W):
	return tf.nn.conv2d(X,W,strides=[1,1,1,1],padding='SAME')
def max_pool(X):
	return tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
X=tf.placeholder(tf.float32,shape=[None,784])
y_=tf.placeholder(tf.float32,shape=[None,10])

X_image=tf.reshape(X,shape=[-1,28,28,1])
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(X_image,W_conv1)+b_conv1)
h_pool1=max_pool(h_conv1)

W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool(h_conv2)

W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1, 7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

y=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
print y

cross_entropy=-tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
correct_pred=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
acc=tf.reduce_mean(tf.cast(correct_pred, "float"))
for i in range(1001):
	batch=mnist.train.next_batch(50)
	if i%10 == 0:
		train_acc=acc.eval(feed_dict={X:batch[0], y_:batch[1],keep_prob:1.0})
		print "step %d, training accuracy %g"%(i,train_acc)
	train_step.run(feed_dict={X:batch[0],y_:batch[1],keep_prob:0.5})
batch=mnist.train.next_batch(5000)
print "test accuracy %g"%acc.eval(feed_dict={X:batch[0],y_:batch[1],keep_prob:1.0})





