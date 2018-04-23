# -*- coding: utf-8 -*-
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100
#学习率和学习衰减率
LEARING_RATA_BASE = 0.8
LEARING_RATA_DELAY = 0.99
#正则化系数、训练轮数、滑动平均衰减率
REGUILARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DELAY = 0.99
#模型保存的路径和文件名
MODEL_SAVE_PATH = "/model/MnistModel/"
MODEL_NAME = "MnistModel.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE], name = 'x-input')
    y_ = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGUILARIZATION_RATE)

    y = mnist_inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DELAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #损失函数
    loss = cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    #loss = cross_entropy_mean
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARING_RATA_BASE,global_step,
                    mnist.train.num_examples/BATCH_SIZE,LEARING_RATA_DELAY)
    # 自动更新global_step
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
                 loss,global_step=global_step)
    #一起执行train_step和variable_averages_op
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

    #初始化TensorFlow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],feed_dict={x:xs,y_:ys})

            if i % 1000 == 0:
                print("After %d training step, loss on training batch is %g"%(step,loss_value))
                #print((sess.run(tf.argmax(y, 1), feed_dict={x: xs})))
                #print((sess.run(tf.argmax(y_, 1), feed_dict={x: xs,y_: ys})))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()