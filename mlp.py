import math

import tensorflow as tf
import numpy as np

learning_rate = 0.0001
display_step = 1

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])  # tf.matmul矩阵乘法，然后各个维度加上偏置
    layer_1 = tf.nn.leaky_relu(layer_1)  # 第一个隐层的输出
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.leaky_relu(layer_2)  # 第二个隐层的输出

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.leaky_relu(layer_3)  # 第三个隐层的输出
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    out_layer = tf.nn.softmax(out_layer)
    return out_layer


class MLP_network:
    def __init__(self):
        # Network Parameters网络结构
        self.n_input = 608
        self.n_hidden_1 = 1024  # 1st layer number of features
        self.n_hidden_2 = 512  # 2nd layer number of features
        self.n_hidden_3 = 256
        self.n_classes = 70
        # tf Graph input 输入
        self.x = tf.placeholder("float", [None, self.n_input])  # 占位，输入维度和数据类型
        self.y = tf.placeholder("float", [None, self.n_classes])  # 占位，输出维度和数据类型
        # Store layers weight & bias 权重和偏置
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]) * math.sqrt(2 / self.n_input)),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]) * math.sqrt(2 / self.n_hidden_1)),
            'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3]) * math.sqrt(2 / self.n_hidden_2)),
            'out': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_classes]) * math.sqrt(2 / self.n_hidden_3))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([self.n_hidden_3])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
        self.pred = multilayer_perceptron(self.x, self.weights, self.biases)
        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost, var_list=[self.biases,
                                                                                                           self.weights])
        print("defining the MLP")

    def train(self, data, label, sess):
        # Initializing the variables
        batch_size = 50
        for i in range(0, 2):
            print(i)
            _, c = sess.run([self.optimizer, self.cost],
                            feed_dict={self.x: data[i * batch_size:(i + 1) * batch_size - 1],
                                       self.y: label[i * batch_size:(i + 1) * batch_size - 1]})
            print('loss:', c)
            print('================================================================================')
        pp = sess.run([self.pred], feed_dict={self.x: data, self.y: label})[0]
        cc = []
        for aa, bb in enumerate(pp[0]):
            if bb != 0:
                cc.append(aa)
        print(cc)
        return pp
