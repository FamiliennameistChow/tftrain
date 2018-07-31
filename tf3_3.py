#coding:utf-8
#simple NN (Fully connected)
import tensorflow as tf

#defind the input and para
x = tf.constant([[0.7,0.5]])
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#defind the Forward propagation
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#calculation with Session
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print("y is:",sess.run(y))

