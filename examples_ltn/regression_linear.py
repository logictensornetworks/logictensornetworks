# -*- coding: utf-8 -*-
import logging; logging.basicConfig(level=logging.DEBUG)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import logictensornetworks as ltn
import logictensornetworks_library as ltnl

start=-1
end=1
training_size=10
testing_size=10
learning_rate = 0.01
slope=1.
var=0.001
epochs=1000

train_X = np.random.uniform(start,end,(training_size)).astype("float32")
train_Y = slope*train_X + np.random.normal(scale=var,size=len(train_X))

W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")
def apply_fun(X):
    return tf.add(tf.multiply(X, W), b)

c_x=[ltn.constant("x_%s" % i,[x]) for i,x in enumerate(train_X)]
c_y=[ltn.constant("y_%s" % i,[y]) for i,y in enumerate(train_Y)]

f=ltn.function("f",1,1,fun_definition=apply_fun)
eq=ltn.predicate("equal",2,lambda x,y: ltnl.equal_euclidian(x,y))

facts=[eq(f(x),y) for x,y in zip(c_x,c_y)]
cost=-tf.reduce_mean(tf.stack(facts))

sess=tf.Session()
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess.run(init)
for i in range(epochs):
    sess.run(opt)
    if i % 10 == 0:
        print(i,"sat level -----> ",sess.run(cost))

# Testing example
x=ltn.variable("x",1)
plt.figure()
plt.plot(train_X, train_Y, 'bo', label='Training data',color="black")
plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
plt.plot(train_X, sess.run(f(x),feed_dict={x:train_X.reshape(len(train_X),1)}), 'bo', label='prediction',color="red")
plt.legend()
plt.show()

test_X = np.random.uniform(start,end,(testing_size)).astype("float32")
test_Y = slope*train_X + np.random.normal(scale=var,size=len(train_X))
plt.figure()
plt.plot(test_X, test_Y, 'bo', label='Testing data')
plt.plot(test_X, sess.run(f(x),feed_dict={x:test_X.reshape(len(test_X),1)}), 'bo', label='prediction',color="red")
plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
plt.legend()
plt.show()