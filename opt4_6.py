# coding:utf-8
import tensorflow as tf

# defined var and ema
w1 = tf.Variable(0, dtype=tf.float32)
# defined num_updates(the turn of NN)
global_step = tf.Variable(0, trainable=False)
# defined the ema 
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)


ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run([w1, ema.average(w1)]))

    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    
    for i in range(40):   
        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))


