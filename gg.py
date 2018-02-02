import tensorflow as tf
x=tf.Variable(3)
y=x*5
sess=tf.Session()
sess.run(tf.global_variables_initializer())
print sess.run(y)
