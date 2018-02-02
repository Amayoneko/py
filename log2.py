from sklearn.datasets import make_moons
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import tensorflow as tf
import sklearn

m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons,X_moons**2,X_moons**3,X_moons[:,0]*X_moons[:,1]]
y_moons_column_vector = y_moons.reshape(-1, 1)
test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]
def random_batch(X_train, y_train, batch_size):
	rnd_indices = np.random.randint(0, len(X_train), batch_size)
	X_batch = X_train[rnd_indices]
	y_batch = y_train[rnd_indices]
	return X_batch, y_batch

n_inputs = 2+5
X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
#theta = tf.Variable(tf.random_uniform((n_inputs + 1, 1), -1.0, 1.0, seed=42), name="theta")
theta = tf.Variable(tf.constant((1.,0.,0.),shape=[n_inputs+1,1]), name="theta")
logits = tf.matmul(X, theta)
#y_proba = 1 / (1 + tf.exp(-logits))
y_proba = tf.sigmoid(logits)
epsilon = 1e-7 
loss = -tf.reduce_mean(y * tf.log(y_proba + epsilon) + (1 - y) * tf.log(1 - y_proba + epsilon))
#loss = tf.losses.log_loss(y, y_proba)

learning_rate = 100
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op =optimizer.minimize(loss)
init = tf.global_variables_initializer()
n_epochs = 1000
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

sess=tf.Session()
sess.run(init)
for epoch in range(n_epochs):
	for batch_index in range(n_batches):
		X_batch, y_batch = random_batch(X_train, y_train, batch_size)
		sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
	loss_val = sess.run(loss,{X: X_test, y: y_test})
	if epoch % 100 == 0:
		print("Epoch:", epoch, "\tLoss:", loss_val)
	"""
	for XX in range(1000):
		sess.run(training_op, feed_dict={X: X_train, y: y_train})
	"""
y_proba_val = sess.run(y_proba,feed_dict={X: X_test, y: y_test})
y_pred = (y_proba_val >= 0.5)
print precision_score(y_test, y_pred)
print recall_score(y_test, y_pred)





