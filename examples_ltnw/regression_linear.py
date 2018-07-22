# -*- coding: utf-8 -*-
import logging; logging.basicConfig(level=logging.DEBUG)
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
var=0.1
epochs=1000

# define data
train_X = np.random.uniform(start,end,(training_size)).astype("float32")
train_Y = slope*train_X + np.random.normal(scale=var,size=len(train_X))

# define the language, we translate every training example into constants
[ltnw.constant("x_%s" % i,[x]) for i,x in enumerate(train_X)]
[ltnw.constant("y_%s" % i,[y]) for i,y in enumerate(train_Y)]

# define the function f as a linear regressor
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")
ltnw.function("f",1,1,fun_definition=lambda X: tf.add(tf.multiply(X, W), b))

# defining an equal predicate based on the euclidian distance of two vectors
ltnw.predicate("eq",2,ltnl.equal_euclidian)

# defining the theory
for f in ["eq(f(x_%s),y_%s)" % (i,i) for i in range(len(train_X))]:
    ltnw.axiom(f)
print("\n".join(sorted(ltnw.AXIOMS.keys())))

# initializing knowledgebase and optimizing
ltnw.initialize_knowledgebase(optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate))
ltnw.train(max_epochs=epochs)

# visualize results on training data
ltnw.variable("?x",1)
prediction=ltnw.ask("f(?x)",feed_dict={"?x" : train_X.reshape(len(train_X),1)})
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_X, train_Y, 'bo', label='Training data',color="black")
plt.plot(train_X, ltnw.SESSION.run(W) * train_X + ltnw.SESSION.run(b), label='Fitted line')
plt.plot(train_X, prediction, 'bo', label='prediction',color="red")
plt.legend()

# generate test data and visualize regressor results
test_X = np.random.uniform(start,end,(testing_size)).astype("float32")
prediction=ltnw.ask("f(?x)",feed_dict={"?x" : test_X.reshape(len(test_X),1)})
test_Y = slope*test_X + np.random.normal(scale=var,size=len(train_X))
plt.subplot(1,2,2)
plt.plot(test_X, test_Y, 'bo', label='Testing data')
plt.plot(test_X, prediction, 'bo', label='prediction',color="red")
plt.plot(test_X, ltnw.SESSION.run(W) * test_X + ltnw.SESSION.run(b), label='Fitted line')
plt.legend()
plt.show()