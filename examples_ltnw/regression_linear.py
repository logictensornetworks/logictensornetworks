# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import logictensornetworks_wrapper as ltnw
import logictensornetworks_library as ltnl

start=-1
end=1
training_size=10
testing_size=10
learning_rate = 0.01
slope=1.
var=0.001
epochs=10000

train_X = np.random.uniform(start,end,(training_size)).astype("float32")
train_Y = slope*train_X + np.random.normal(scale=var,size=len(train_X))

W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")
def apply_fun(X):
    return tf.add(tf.multiply(X, W), b)

[ltnw.constant("x_%s" % i,[x]) for i,x in enumerate(train_X)]
[ltnw.constant("y_%s" % i,[y]) for i,y in enumerate(train_Y)]

ltnw.function("f",1,1,fun_definition=apply_fun)
ltnw.predicate("eq",2,lambda x,y: ltnl.equal_euclidian(x,y))

formulas=["eq(f(x_%s),y_%s)" % (i,i) for i in range(len(train_X))]
print("\n".join(formulas))
for f in formulas:
    ltnw.formula(f)

ltnw.initialize_knowledgebase(optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate))
ltnw.train(max_iterations=epochs)

# Testing example
x=ltnw.variable("x",1)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_X, train_Y, 'bo', label='Training data',color="black")
plt.plot(train_X, ltnw.SESSION.run(W) * train_X + ltnw.SESSION.run(b), label='Fitted line')
plt.plot(train_X, ltnw.ask("f(x)",feed_dict={"x":train_X.reshape(len(train_X),1)}), 'bo', label='prediction',color="red")
plt.legend()

test_X = np.random.uniform(start,end,(testing_size)).astype("float32")
test_Y = slope*train_X + np.random.normal(scale=var,size=len(train_X))
plt.subplot(1,2,2)
plt.plot(test_X, test_Y, 'bo', label='Testing data')
plt.plot(test_X, ltnw.ask("f(x)",feed_dict={x:test_X.reshape(len(test_X),1)}), 'bo', label='prediction',color="red")
plt.plot(train_X, ltnw.SESSION.run(W) * train_X + ltnw.SESSION.run(b), label='Fitted line')
plt.legend()
plt.show()