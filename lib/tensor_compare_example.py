import numpy as np
import tensorflow as tf

with tf.Session() as sess:
    a = np.zeros((2,2))
    ta = tf.zeros((2,2))
    print(a)
    print(ta)
    print(ta.eval())

W = tf.Variable(tf.zeros((2,2)), name="weights")
R = tf.Variable(tf.random_normal((2,2)), name="random_weights")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(W))
    print(sess.run(R))
    print(R.eval())

x = 3.0
y = 3.0

result1 = tf.equal(x, y)

# Use reduce_all and reduce_any to test the results of equal.
result2 = tf.reduce_all(result1)

with tf.Session() as sess:
    print(result1.eval())
    print(result2.eval())

c = tf.Variable(5.00000, tf.int64)

# https://www.tensorflow.org/api_docs/python/tf/cond
y = tf.cond(tf.equal(c, 5), lambda: tf.add(c, 1.0), lambda: tf.add(c, -1.0))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(y.eval())

