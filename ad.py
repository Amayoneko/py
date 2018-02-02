import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=(None, None))  
y = tf.log(x)
         
with tf.Session() as sess:  
      #print(sess.run(y))
      rand_array = np.random.rand(1023, 1024)  
      print(sess.run(y, feed_dict={x: rand_array})) 
print(rand_array)
print type(x)
