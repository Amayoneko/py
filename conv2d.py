import tensorflow as tf
 
data=tf.Variable(tf.random_normal([64,48,48,3]),dtype=tf.float32)
weight=tf.Variable(tf.random_normal([5,5,3,65]),dtype=tf.float32)
 
sess=tf.Session()
sess.run(tf.global_variables_initializer())
 
conv1=tf.nn.conv2d(data,weight,strides=[1,1,1,1],padding='SAME')
conv2=tf.nn.conv2d(data,weight,strides=[1,5,5,1],padding='SAME')
conv3=tf.nn.conv2d(data,weight,strides=[1,4,4,1],padding='SAME')
 
print(conv1)
print(conv2)
print(conv3)
