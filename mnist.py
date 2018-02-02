import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
X=tf.placeholder(tf.float32,shape=[785,None])
W=tf.Variable(tf.zeros(shape=[10,785]))
Y=tf.nn.softmax(tf.matmul(W,X))
Y_=tf.placeholder(tf.float32,shape=[10,None])
cross_entropy=-tf.reduce_sum(Y_*tf.log(Y))
np.set_printoptions(threshold='nan')
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
comp=tf.equal(tf.argmax(Y,0),tf.argmax(Y_,0))
rate=tf.reduce_mean(tf.cast(comp,tf.float32))
batch_X=mnist.train.images
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	batch_size=100
	for times in range(int(60000/batch_size)):
		batch_X,batch_Y=mnist.train.next_batch(batch_size)
		#batch_X=np.float32(batch_X!=0)
		batch_X=batch_X.T
		batch_X=np.r_[batch_X,np.ones(shape=[1,batch_size])]
		batch_Y=batch_Y.T;
		sess.run(train_step,feed_dict={X:batch_X,Y_:batch_Y})
	a=W.eval()
	batch_size=10000
	batch_X,batch_Y=mnist.test.next_batch(batch_size)
	#batch_X=np.float32(batch_X!=0)
	batch_X=batch_X.T
	batch_X=np.r_[batch_X,np.ones(shape=[1,batch_size])]
	batch_Y=batch_Y.T;
	print sess.run(rate,feed_dict={X:batch_X,Y_:batch_Y})
	C=sess.run(comp,feed_dict={X:batch_X,Y_:batch_Y})
	hit=[0]*10
	ttl=[0]*10
	for p in range(len(C)):
		t=np.argmax(batch_Y[:,p])
		if(C[p]==True):hit[t]+=1
		ttl[t]+=1
	for p in range(10):
		print p,hit[p]*1.0/ttl[p]

fig,ax=plt.subplots(2,5)
ax=ax.flatten()
np.set_printoptions(precision=2,linewidth=150)
for p in range(10):
	b=a[p]
	ax[p].imshow(b[:784].reshape(28,28),cmap="Greys")
	b=b[:784]*100
	b=b.astype(np.int32)
	#print b.reshape(28,28)
plt.show()



