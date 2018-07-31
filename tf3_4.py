#coding:utf-8
#simple NN

import tensorflow as tf

#defind input and para
#sue the placeholder to input
x = tf.placeholder(tf.float32,shape=(1,2))
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#defind teh Forward propagation
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print("y is:", sess.run(y, feed_dict={x: [[0.5, 0.6]]}))

