# coding:utf-8
# set the loss = (w+1)^2 set the defult value of the w is 5
import tensorflow as tf
# set para w as 5
w = tf.Variable(tf.constant(5, dtype=tf.float32))
# set the loss
loss = tf.square(w + 1)
# set the BP
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s steps: w is %f,   loss is %f" % (i, w_val, loss_val))
