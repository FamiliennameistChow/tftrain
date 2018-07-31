# coding:utf-8
# set the loss = (w+1)^2 set the defult value of the w is 5
import tensorflow as tf

LEARNING_RATE_BASE = 0.1 # the defult learning rate
LEARNING_RATE_DECAY = 0.99 # the decay of the learing rate
LEARNING_RATE_STEP =1 # how much turns to renewal learning_rate

# the counter of turns BATACH_STEP
global_step = tf.Variable(0, trainable=False)

# defined the learning_rate
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)
# set para w as 5
w = tf.Variable(tf.constant(5, dtype=tf.float32))
# set the loss
loss = tf.square(w + 1)
# set the BP
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step= global_step)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        print("After %s steps: global_step is %f, learning_rate is %f, w is %f,   loss is %f" % (i, global_step_val, learning_rate_val, w_val, loss_val))
