from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

minst = input_data.read_data_sets('MINST_data/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    xs, ys = minst.train.next_batch(100)
    sess.run(train_step, feed_dict={x: xs, y_: ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: minst.test.images, y_: minst.test.labels}))
