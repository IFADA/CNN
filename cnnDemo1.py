import  tensorflow  as tf
import matplotlib.pyplot as plt
import  matplotlib.image as mping
import  numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
class Net:
  def __init__(self):
      self.x = tf.placeholder(dtype=tf.float32,shape=[None,28,28,1])
      self.y = tf.placeholder(dtype=tf.float32,shape=[None,10])
      self.dp = tf.placeholder(dtype=tf.float32)
      self.conw1 = tf.Variable(tf.random_normal(shape=[3,3,1,16],stddev=0.1))
      self.conb1 = tf.Variable(tf.zeros([16]))
  def forward(self):
      self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x,self.conw1,strides=[1,1,1,1],padding='SAME')+self.conb1)
      self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
      self.conw2 = tf.Variable(tf.random_normal(shape=[3,3,16,32],stddev=0.1))
      self.conb2 = tf.Variable(tf.zeros([32]))

      self.conv2 = tf.nn.relu(tf.nn.conv2d(self.pool1,self.conw2,strides=[1,1,1,1],padding='SAME')+self.conb2)
      self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
      #卷积后的形状
      self.reshape_ = tf.reshape(self.pool2,[-1,7*7*32])
      self.w1 = tf.Variable(tf.random_normal(shape=[7*7*32,128],stddev=0.1))
      self.b1 = tf.Variable(tf.zeros([128]))
      self.fcl = tf.nn.relu(tf.matmul(self.reshape_,self.w1)+self.b1)

      self.w2 = tf.Variable(tf.random_normal(shape=[128,10],stddev=0.1))
      self.b2 = tf.Variable(tf.zeros([10]))
      self.drop_ = tf.nn.dropout(self.fcl,keep_prob=self.dp)
      self.output = tf.nn.relu(tf.matmul(self.drop_,self.w2)+self.b2)

  def backward(self):
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.output,logits=self.y))
      self.opt = tf.train.AdamOptimizer().minimize(self.loss)
      self.predict = tf.equal(tf.argmax(self.output,axis=1),tf.argmax(self.y,axis=1))
      self.rs = tf.cast(self.predict,'float')
      self.accurate = tf.reduce_mean(self.rs)
if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        xs,ys = mnist.test.next_batch(100)
        xs_xs = xs.reshape([100,28,28,1])
        _loss,_,_ = sess.run([net.loss,net.opt,net.accurate],feed_dict={net.x:xs_xs,net.y:ys,net.dp:0.9})
