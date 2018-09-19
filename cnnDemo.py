import tensorflow  as tf
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
class Net:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.dp = tf.placeholder(tf.float32)
        self.conv1_w = tf.Variable(tf.random_normal([3, 3, 1, 16], dtype=tf.float32, stddev=0.1))
        self.conv1_b = tf.Variable(tf.zeros([16]))

    def forward(self):
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x, self.conv1_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv1_b)
        self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.conv2_w = tf.Variable(tf.random_normal([3, 3, 16, 32], dtype=tf.float32, stddev=0.1))
        self.conv2_b = tf.Variable(tf.zeros([32]))

        self.conv2 = tf.nn.relu(
            tf.nn.conv2d(self.pool1, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b)
        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.flat = tf.reshape(self.pool2, [-1, 7 * 7 * 32])

        self.w1 = tf.Variable(tf.random_normal([7 * 7 * 32, 128], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([128]))

        self.fcl1 = tf.nn.relu(tf.matmul(self.flat, self.w1) + self.b1)

        self.w2 = tf.Variable(tf.random_normal([128, 10], stddev=0.1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros([10]))

        self.drop = tf.nn.dropout(self.fcl1, keep_prob=self.dp)
        self.output = tf.nn.softmax(tf.matmul(self.drop, self.w2) + self.b2)

    def backward(self):
        # cross_entropy =tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)#这个是误差方程
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))

        self.optimizer = tf.train.AdamOptimizer(0.003)
        self.train_step = self.optimizer.minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.rst = tf.cast(self.correct_prediction, "float")  # 数据类型转换
        self.accuracy = tf.reduce_mean(self.rst)


if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    # saver =tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(100000):
            bach_x_xs, bach_y = mnist.train.next_batch(128)
            bach_x = bach_x_xs.reshape([128, 28, 28, 1])
            _, loss, acc = sess.run([net.train_step, net.cross_entropy, net.accuracy],
                                    feed_dict={net.x: bach_x, net.y: bach_y, net.dp: 0.9})

            batch_xs_x, batch_ys = mnist.test.next_batch(128)
            batch_xs = batch_xs_x.reshape([128, 28, 28, 1])  # 这个地方要注意，传入参数满足N,H,W,C的传参标准

            if epoch % 100 == 0:
                print("精度：{0}".format(acc))
