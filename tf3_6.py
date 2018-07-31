#coding:utf-8
#import modlue
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455

#produce random num with seed 
rng = np.random.RandomState(seed)
# use the random num to return a matrix of 32*2
X = rng.rand(32,2)
# take a line out of the matrix ,judege the sum ,if it is >1,return Y=1,if else, return Y=0
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print("X:\n", X)
print("Y:\n", Y)
 
#defind the input,parameter,output and the process of the forward propagation
x = tf.placeholder(tf.float32, shape=(None, 2))
y_= tf.placeholder(tf.float32, shape=(None, 1))

w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#defind the loss and BP
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimizer(loss)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	# output the untrained para
	print("w1:\n", sess.run(w1))
	print("w2:\n", sess.run(w2))
	print("\n")

	# training 
	STEPS = 3000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
		if i % 500 == 0:
			total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
			print("After %d teaining steps, loss on all data is %g" % (i, total_loss))

	# output the trained para
	print('\n')
	print("w1:\n", sess.run(w1))
	print("w2:\n", sess.run(w2))

 
